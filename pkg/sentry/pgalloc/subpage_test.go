// Copyright 2026 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pgalloc

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"slices"
	"testing"
	"time"

	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/memutil"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
	pgallocpb "gvisor.dev/gvisor/pkg/sentry/pgalloc/pgalloc_metadata_go_proto"
	"gvisor.dev/gvisor/pkg/sentry/state/stateio"
	"gvisor.dev/gvisor/pkg/timing"
)

func TestForEachSubpageRun(t *testing.T) {
	for _, tc := range []struct {
		mask uint64
		want [][2]uint64
	}{
		{0, nil},
		{0b1, [][2]uint64{{0, 1}}},
		{0b1011_0110, [][2]uint64{{1, 3}, {4, 6}, {7, 8}}},
		{^uint64(0), [][2]uint64{{0, 64}}},
		{1 << 63, [][2]uint64{{63, 64}}},
		{^uint64(0) &^ (1 << 32), [][2]uint64{{0, 32}, {33, 64}}},
	} {
		var got [][2]uint64
		forEachSubpageRun(tc.mask, func(first, end uint64) { got = append(got, [2]uint64{first, end}) })
		if !slices.Equal(got, tc.want) {
			t.Errorf("forEachSubpageRun(%#b) = %v, want %v", tc.mask, got, tc.want)
		}
	}
}

// validSubpageSizes returns the valid subpage sizes for this build's page
// size, in increasing order.
func validSubpageSizes() []uint64 {
	var sizes []uint64
	for size := uint64(1); size < hostarch.PageSize; size *= 2 {
		if ValidSubpageSize(size) {
			sizes = append(sizes, size)
		}
	}
	return sizes
}

func TestValidSubpageSize(t *testing.T) {
	for _, tc := range []struct {
		size uint64
		want bool
	}{
		{0, false},
		{256, false},
		{511, false},
		{513, false},
		{3000, false},
		{hostarch.PageSize, false},
		{2 * hostarch.PageSize, false},
		{hostarch.PageSize / 2, true},
		// Too many blocks per page on hosts with pages larger than 32 KiB.
		{512, hostarch.PageSize <= 32<<10},
	} {
		if got := ValidSubpageSize(tc.size); got != tc.want {
			t.Errorf("ValidSubpageSize(%d) = %t, want %t", tc.size, got, tc.want)
		}
	}
	sizes := validSubpageSizes()
	if len(sizes) == 0 || hostarch.PageSize/sizes[0] > 64 {
		t.Errorf("valid subpage sizes %v for page size %d", sizes, hostarch.PageSize)
	}
	t.Logf("valid subpage sizes for page size %d: %v", hostarch.PageSize, sizes)

	size4K := SubpageSize4K()
	want4K := uint64(0)
	if hostarch.PageSize > 4096 && hostarch.PageSize/4096 <= 64 {
		want4K = 4096
	}
	if size4K != want4K {
		t.Errorf("SubpageSize4K() = %d, want %d for page size %d", size4K, want4K, hostarch.PageSize)
	}
}

func TestMaxSubpageRuns(t *testing.T) {
	for _, size := range validSubpageSizes() {
		n := 0
		forEachSubpageRun(subpageAltMask(size), func(first, end uint64) { n++ })
		if got := maxSubpageRuns(size); got != n {
			t.Errorf("maxSubpageRuns(%d) = %d, want %d", size, got, n)
		}
	}
}

// subpageAltMask returns the mask of every other block of a page, starting
// with the first, which has the most runs of any mask.
func subpageAltMask(size uint64) uint64 {
	return 0x5555_5555_5555_5555 & fullSubpageMask(size)
}

// subpageTestMasks returns the masks of non-zero size-byte blocks in each page
// of test contents: single blocks at either end of a page, maximally
// fragmented pages, full pages, several runs, and zero pages.
func subpageTestMasks(size uint64) []uint64 {
	n := hostarch.PageSize / size
	full := fullSubpageMask(size)
	alt := subpageAltMask(size)
	return []uint64{
		1,
		alt,
		full,
		0,
		1 << (n - 1),
		(alt << 1) & full,
		0,
		0,
		1 | 1<<(n/2) | 1<<(n-1),
		full,
	}
}

// dstRangeReader is a stateio.AsyncReader that, like the stateipc client used
// with a checkpoint gofer, reads into registered destination files at the
// destination FileRanges it is given, ignoring mappings. Reads complete
// synchronously.
type dstRangeReader struct {
	fd        int
	maxRanges int
	dsts      []int
	done      []stateio.Completion
}

func (r *dstRangeReader) Close() error {
	for _, fd := range r.dsts {
		unix.Close(fd)
	}
	return unix.Close(r.fd)
}

func (r *dstRangeReader) MaxReadBytes() uint64            { return 16 * hostarch.PageSize }
func (r *dstRangeReader) MaxRanges() int                  { return r.maxRanges }
func (r *dstRangeReader) MaxParallel() int                { return 4 }
func (r *dstRangeReader) NeedRegisterDestinationFD() bool { return true }

func (r *dstRangeReader) RegisterDestinationFD(fd int32, size uint64, settings []stateio.ClientFileRangeSetting) (stateio.DestinationFile, error) {
	dfd, err := unix.Dup(int(fd))
	if err != nil {
		return nil, err
	}
	r.dsts = append(r.dsts, dfd)
	return dfd, nil
}

func (r *dstRangeReader) AddRead(id int, off int64, dstFile stateio.DestinationFile, dstFR memmap.FileRange, dstMap []byte) {
	if uint64(len(dstMap)) != dstFR.Length() {
		r.done = append(r.done, stateio.Completion{ID: id, Err: fmt.Errorf("AddRead: %d-byte mapping of %v", len(dstMap), dstFR)})
		return
	}
	r.read(id, off, dstFR.Length(), dstFile, []memmap.FileRange{dstFR})
}

func (r *dstRangeReader) AddReadv(id int, off int64, total uint64, dstFile stateio.DestinationFile, dstFRs []memmap.FileRange, dstMaps []unix.Iovec) {
	if len(dstFRs) > r.maxRanges || len(dstMaps) > r.maxRanges {
		r.done = append(r.done, stateio.Completion{ID: id, Err: fmt.Errorf("AddReadv: %d ranges and %d iovecs, limit %d", len(dstFRs), len(dstMaps), r.maxRanges)})
		return
	}
	r.read(id, off, total, dstFile, dstFRs)
}

func (r *dstRangeReader) read(id int, off int64, total uint64, dstFile stateio.DestinationFile, dstFRs []memmap.FileRange) {
	c := stateio.Completion{ID: id}
	for _, fr := range dstFRs {
		buf := make([]byte, fr.Length())
		n, err := unix.Pread(r.fd, buf, off+int64(c.N))
		if err == nil && n < len(buf) {
			err = io.ErrUnexpectedEOF
		}
		if err == nil {
			_, err = unix.Pwrite(dstFile.(int), buf, int64(fr.Start))
		}
		if err != nil {
			c.Err = err
			break
		}
		c.N += uint64(n)
	}
	if c.Err == nil && c.N != total {
		c.Err = fmt.Errorf("read %d bytes into %v, want %d", c.N, dstFRs, total)
	}
	r.done = append(r.done, c)
}

func (r *dstRangeReader) Wait(cs []stateio.Completion, minCompletions int) ([]stateio.Completion, error) {
	cs = append(cs, r.done...)
	r.done = r.done[:0]
	return cs, nil
}

// newTestMemoryFile returns a new MemoryFile backed by a temporary file if
// diskBacked is true, or by a memfd otherwise. The caller must destroy it.
func newTestMemoryFile(tb testing.TB, diskBacked bool) *MemoryFile {
	tb.Helper()
	var file *os.File
	if diskBacked {
		var err error
		if file, err = os.CreateTemp(tb.TempDir(), "pgalloc-subpage-test"); err != nil {
			tb.Fatalf("CreateTemp: %v", err)
		}
		os.Remove(file.Name())
	} else {
		fd, err := memutil.CreateMemFD("pgalloc-subpage-test", 0)
		if err != nil {
			tb.Fatalf("CreateMemFD: %v", err)
		}
		file = os.NewFile(uintptr(fd), "pgalloc-subpage-test")
	}
	f, err := NewMemoryFile(file, MemoryFileOpts{
		DelayedEviction:         DelayedEvictionDisabled,
		DisableMemoryAccounting: true,
		DiskBackedFile:          diskBacked,
	})
	if err != nil {
		file.Close()
		tb.Fatalf("NewMemoryFile: %v", err)
	}
	return f
}

// subpageTestFile is a MemoryFile whose contents are described by a mask of
// non-zero blocks per page.
type subpageTestFile struct {
	f          *MemoryFile
	diskBacked bool
	fr         memmap.FileRange
	want       []byte

	// subpageSize is the SaveOpts.SubpageSize the file is saved with.
	subpageSize uint64

	// nonzeroBlocks is the number of non-zero blocks of the size masks were
	// given in, and nonzeroPages is the number of non-zero pages.
	nonzeroBlocks uint64
	nonzeroPages  uint64
}

// newSubpageTestFile returns a MemoryFile whose i-th page has non-zero bytes
// in the size-byte blocks set in masks[i], to be saved with subpageSize.
func newSubpageTestFile(tb testing.TB, diskBacked bool, size uint64, masks []uint64, subpageSize uint64) *subpageTestFile {
	tb.Helper()
	tf := &subpageTestFile{
		f:           newTestMemoryFile(tb, diskBacked),
		diskBacked:  diskBacked,
		subpageSize: subpageSize,
	}
	tb.Cleanup(tf.f.Destroy)
	fr, err := tf.f.Allocate(uint64(len(masks))*hostarch.PageSize, AllocOpts{Mode: AllocateUncommitted})
	if err != nil {
		tb.Fatalf("Allocate: %v", err)
	}
	tb.Cleanup(func() { tf.f.DecRef(fr) })
	tf.fr = fr
	tf.want = make([]byte, fr.Length())
	for p, mask := range masks {
		forEachSubpageRun(mask, func(first, end uint64) {
			b := tf.want[uint64(p)*hostarch.PageSize+first*size : uint64(p)*hostarch.PageSize+end*size]
			for i := range b {
				b[i] = byte(1 + (p*31+i)%255)
			}
			tf.nonzeroBlocks += end - first
		})
		if mask != 0 {
			tf.nonzeroPages++
		}
	}
	if _, err := unix.Pwrite(tf.f.FD(), tf.want, int64(fr.Start)); err != nil {
		tb.Fatalf("Pwrite: %v", err)
	}
	return tf
}

// newPagesFile returns an empty pages file.
func newPagesFile(tb testing.TB) *os.File {
	tb.Helper()
	pages, err := os.CreateTemp(tb.TempDir(), "pages")
	if err != nil {
		tb.Fatalf("CreateTemp: %v", err)
	}
	tb.Cleanup(func() { pages.Close() })
	return pages
}

func dupFD(tb testing.TB, f *os.File) int {
	tb.Helper()
	fd, err := unix.Dup(int(f.Fd()))
	if err != nil {
		tb.Fatalf("Dup: %v", err)
	}
	return fd
}

// saveAsync saves tfs in order to pages with async page saving, and returns
// their metadata.
func saveAsync(tb testing.TB, pages *os.File, tfs ...*subpageTestFile) *bytes.Buffer {
	tb.Helper()
	meta, _ := saveAsyncRecorded(tb, pages, tfs...)
	return meta
}

// savedMemoryFile locates a MemoryFile saved by saveAsyncRecorded, as a
// filesystem checkpoint manifest does.
type savedMemoryFile struct {
	metaStart, metaEnd int
	pagesStart         uint64
}

// saveAsyncRecorded is saveAsync that also returns where each MemoryFile's
// metadata and pages were saved.
func saveAsyncRecorded(tb testing.TB, pages *os.File, tfs ...*subpageTestFile) (*bytes.Buffer, []savedMemoryFile) {
	tb.Helper()
	saveDone := make(chan error, 1)
	apfs, err := StartAsyncPagesFileSave(stateio.NewPagesFileFDWriterDefault(int32(dupFD(tb, pages))), func(err error) { saveDone <- err })
	if err != nil {
		tb.Fatalf("StartAsyncPagesFileSave: %v", err)
	}
	var (
		meta  bytes.Buffer
		saved []savedMemoryFile
	)
	for _, tf := range tfs {
		s := savedMemoryFile{metaStart: meta.Len(), pagesStart: apfs.PagesFileOffset()}
		if err := tf.f.SaveTo(context.Background(), &meta, &SaveOpts{PagesFile: apfs, SubpageSize: tf.subpageSize}); err != nil {
			apfs.MemoryFilesDone()
			<-saveDone
			tb.Fatalf("SaveTo: %v", err)
		}
		s.metaEnd = meta.Len()
		saved = append(saved, s)
	}
	apfs.MemoryFilesDone()
	if err := <-saveDone; err != nil {
		tb.Fatalf("async page saving: %v", err)
	}
	return &meta, saved
}

// subpageLoadConfig configures async page loading in tests.
type subpageLoadConfig struct {
	// If dstRanges is not 0, pages are loaded by a dstRangeReader with that
	// many ranges per read.
	dstRanges int

	// throttled is AsyncPagesFileLoadOpts.Throttled.
	throttled bool
}

func (lc subpageLoadConfig) String() string {
	reader := "fd reader"
	if lc.dstRanges != 0 {
		reader = fmt.Sprintf("destination ranges %d", lc.dstRanges)
	}
	if lc.throttled {
		reader += ", throttled"
	}
	return reader
}

func startAsyncLoad(tb testing.TB, pages *os.File, lc subpageLoadConfig) *AsyncPagesFileLoad {
	tb.Helper()
	// Construct only the selected reader: an FDReader starts its workers and
	// takes ownership of rfd when constructed.
	rfd := dupFD(tb, pages)
	var ar stateio.AsyncReader
	if lc.dstRanges != 0 {
		ar = &dstRangeReader{fd: rfd, maxRanges: lc.dstRanges}
	} else {
		ar = stateio.NewPagesFileFDReaderDefault(int32(rfd), 1)
	}
	timer := timing.New("test", time.Now())
	tb.Cleanup(timer.End)
	timeline := timer.Fork("restore")
	tb.Cleanup(timeline.End)
	apfl, err := StartAsyncPagesFileLoad(ar, nil, timeline, AsyncPagesFileLoadOpts{Throttled: lc.throttled})
	if err != nil {
		tb.Fatalf("StartAsyncPagesFileLoad: %v", err)
	}
	return apfl
}

// loadAsync loads MemoryFiles saved by saveAsync from meta and pages, in the
// same order, and checks their contents against tfs. Before waiting for all
// pages to load, it waits for each page range in awaits (page indexes into
// each file) and checks its contents.
func loadAsync(t *testing.T, meta *bytes.Buffer, pages *os.File, lc subpageLoadConfig, awaits [][2]uint64, tfs ...*subpageTestFile) {
	t.Helper()
	apfl := startAsyncLoad(t, pages, lc)
	opts := LoadOpts{PagesFile: apfl}
	restored := make([]*MemoryFile, len(tfs))
	for i, tf := range tfs {
		restored[i] = newTestMemoryFile(t, tf.diskBacked)
		t.Cleanup(restored[i].Destroy)
		if err := restored[i].LoadFrom(context.Background(), meta, &opts); err != nil {
			apfl.MemoryFilesDone()
			t.Fatalf("LoadFrom for file %d: %v", i, err)
		}
		t.Cleanup(func() { restored[i].DecRef(tf.fr) })
		t.Logf("file %d loaded, next pages file offset %#x", i, opts.PagesFileOffset)
	}
	apfl.MemoryFilesDone()
	for i, tf := range tfs {
		for _, pgs := range awaits {
			fr := memmap.FileRange{tf.fr.Start + pgs[0]*hostarch.PageSize, tf.fr.Start + pgs[1]*hostarch.PageSize}
			if amfl := restored[i].asyncPageLoad.Load(); amfl != nil {
				restored[i].IncRef(fr, 0)
				err := amfl.awaitLoad(fr)
				restored[i].DecRef(fr)
				if err != nil {
					t.Fatalf("awaitLoad(%v) for file %d: %v", fr, i, err)
				}
			}
			checkSubpageContents(t, restored[i], fr, tf.want[pgs[0]*hostarch.PageSize:pgs[1]*hostarch.PageSize], fmt.Sprintf("file %d pages %v after awaitLoad", i, pgs))
		}
	}
	for i, tf := range tfs {
		if err := restored[i].AwaitLoadAll(); err != nil {
			t.Fatalf("AwaitLoadAll for file %d: %v", i, err)
		}
		checkSubpageContents(t, restored[i], tf.fr, tf.want, fmt.Sprintf("file %d", i))
	}
}

func checkSubpageContents(t *testing.T, f *MemoryFile, fr memmap.FileRange, want []byte, desc string) {
	t.Helper()
	got := make([]byte, 0, fr.Length())
	f.forEachMappingSlice(fr, func(bs []byte) { got = append(got, bs...) })
	if bytes.Equal(got, want) {
		return
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: restored contents differ from saved contents, first at offset %#x (page %d): got %#x, want %#x", desc, i, uint64(i)/hostarch.PageSize, got[i], want[i])
		}
	}
}

func pagesFileSize(t *testing.T, pages *os.File) uint64 {
	t.Helper()
	st, err := pages.Stat()
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	return uint64(st.Size())
}

// parseMetadata returns the metadata of the first MemoryFile in meta, which
// must have been saved with a pages file.
func parseMetadata(t *testing.T, meta []byte) *pgallocpb.MemoryFileMetadataProto {
	t.Helper()
	if len(meta) < 8 {
		t.Fatalf("metadata is %d bytes", len(meta))
	}
	n := binary.LittleEndian.Uint64(meta)
	var pb pgallocpb.MemoryFileMetadataProto
	if err := proto.Unmarshal(meta[8:8+n], &pb); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}
	return &pb
}

func marshalMetadata(t *testing.T, pb *pgallocpb.MemoryFileMetadataProto) *bytes.Buffer {
	t.Helper()
	data, err := proto.Marshal(pb)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}
	var meta bytes.Buffer
	binary.Write(&meta, binary.LittleEndian, uint64(len(data)))
	meta.Write(data)
	return &meta
}

// subpageTestAwaits are page ranges awaited before all pages are loaded:
// single pages in the middle and at either end, out of order, and ranges
// spanning zero pages.
var subpageTestAwaits = [][2]uint64{{4, 5}, {9, 10}, {0, 1}, {2, 9}, {1, 3}}

func TestSubpageSaveRestoreBlockSizes(t *testing.T) {
	for _, size := range validSubpageSizes() {
		for _, lc := range []subpageLoadConfig{{}, {dstRanges: maxSubpageRuns(size)}, {throttled: true}} {
			t.Run(fmt.Sprintf("%d-byte blocks, %v", size, lc), func(t *testing.T) {
				tf := newSubpageTestFile(t, true /* diskBacked */, size, subpageTestMasks(size), size)
				pages := newPagesFile(t)
				meta := saveAsync(t, pages, tf)
				if got, want := pagesFileSize(t, pages), tf.nonzeroBlocks*size; got != want {
					t.Errorf("pages file size = %d, want %d", got, want)
				}
				pb := parseMetadata(t, meta.Bytes())
				if pb.SubpageSize != size || uint64(len(pb.SubpageMasks)) != tf.nonzeroPages {
					t.Errorf("metadata has subpage size %d and %d masks, want %d and %d", pb.SubpageSize, len(pb.SubpageMasks), size, tf.nonzeroPages)
				}
				loadAsync(t, meta, pages, lc, subpageTestAwaits, tf)
			})
		}
	}
}

func TestSubpageSaveDisabled(t *testing.T) {
	size := uint64(hostarch.PageSize / 2)
	for _, tc := range []struct {
		name        string
		diskBacked  bool
		subpageSize uint64
	}{
		{"disk-backed without subpage size", true, 0},
		{"memfd with subpage size", false, size},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tf := newSubpageTestFile(t, tc.diskBacked, size, subpageTestMasks(size), tc.subpageSize)
			pages := newPagesFile(t)
			meta := saveAsync(t, pages, tf)
			if got, want := pagesFileSize(t, pages), tf.nonzeroPages*hostarch.PageSize; got != want {
				t.Errorf("pages file size = %d, want %d", got, want)
			}
			if pb := parseMetadata(t, meta.Bytes()); pb.SubpageSize != 0 || len(pb.SubpageMasks) != 0 {
				t.Errorf("metadata has subpage size %d and %d masks, want none", pb.SubpageSize, len(pb.SubpageMasks))
			}
			loadAsync(t, meta, pages, subpageLoadConfig{}, subpageTestAwaits, tf)
		})
	}
}

func TestSubpageSaveInvalidSize(t *testing.T) {
	sizes := []uint64{1, 256, 511, 513, 3000, hostarch.PageSize, 2 * hostarch.PageSize}
	if s := uint64(hostarch.PageSize / 128); s >= 512 {
		sizes = append(sizes, s)
	}
	for _, size := range sizes {
		for _, async := range []bool{false, true} {
			t.Run(fmt.Sprintf("%d async=%t", size, async), func(t *testing.T) {
				f := newTestMemoryFile(t, true /* diskBacked */)
				t.Cleanup(f.Destroy)
				opts := SaveOpts{SubpageSize: size}
				if async {
					apfs, err := StartAsyncPagesFileSave(stateio.NewPagesFileFDWriterDefault(int32(dupFD(t, newPagesFile(t)))), func(error) {})
					if err != nil {
						t.Fatalf("StartAsyncPagesFileSave: %v", err)
					}
					defer apfs.MemoryFilesDone()
					opts.PagesFile = apfs
				}
				var meta bytes.Buffer
				if err := f.SaveTo(context.Background(), &meta, &opts); err == nil {
					t.Errorf("SaveTo succeeded with subpage size %d, want error", size)
				}
				if meta.Len() != 0 {
					t.Errorf("SaveTo wrote %d bytes of metadata", meta.Len())
				}
			})
		}
	}
}

func TestSubpageSaveSynchronous(t *testing.T) {
	size := validSubpageSizes()[0]
	tf := newSubpageTestFile(t, true /* diskBacked */, size, subpageTestMasks(size), size)
	var state bytes.Buffer
	if err := tf.f.SaveTo(context.Background(), &state, &SaveOpts{SubpageSize: size}); err != nil {
		t.Fatalf("SaveTo: %v", err)
	}
	if pb := parseMetadata(t, state.Bytes()); pb.SubpageSize != 0 || len(pb.SubpageMasks) != 0 {
		t.Errorf("metadata has subpage size %d and %d masks, want none", pb.SubpageSize, len(pb.SubpageMasks))
	}
	restored := newTestMemoryFile(t, true /* diskBacked */)
	t.Cleanup(restored.Destroy)
	if err := restored.LoadFrom(context.Background(), &state, &LoadOpts{}); err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}
	t.Cleanup(func() { restored.DecRef(tf.fr) })
	checkSubpageContents(t, restored, tf.fr, tf.want, "synchronously loaded file")
}

func TestSubpageSaveRestoreSharedPagesFile(t *testing.T) {
	// Blocks of the smallest valid size leave the pages file offset after a
	// sparse MemoryFile unaligned to both pages and 4 KiB.
	size := validSubpageSizes()[0]
	masks := subpageTestMasks(size)
	for _, lc := range []subpageLoadConfig{{}, {dstRanges: maxSubpageRuns(size)}, {throttled: true}} {
		t.Run(lc.String(), func(t *testing.T) {
			tfs := []*subpageTestFile{
				newSubpageTestFile(t, true /* diskBacked */, size, masks, size),
				newSubpageTestFile(t, false /* diskBacked */, size, masks, size),
				newSubpageTestFile(t, true /* diskBacked */, size, masks, size),
				newSubpageTestFile(t, true /* diskBacked */, size, masks, 0),
				newSubpageTestFile(t, true /* diskBacked */, size, masks, size),
			}
			pages := newPagesFile(t)
			meta := saveAsync(t, pages, tfs...)
			want := uint64(0)
			for _, tf := range tfs {
				if tf.diskBacked && tf.subpageSize != 0 {
					want += tf.nonzeroBlocks * size
				} else {
					want += tf.nonzeroPages * hostarch.PageSize
				}
			}
			if got := pagesFileSize(t, pages); got != want {
				t.Errorf("pages file size = %d, want %d", got, want)
			}
			loadAsync(t, meta, pages, lc, subpageTestAwaits, tfs...)
		})
	}
}

// TestSubpageLoadAtRecordedOffsets loads MemoryFiles individually and in
// reverse order, each from the metadata range and pages file offset recorded
// when it was saved, as filesystem checkpoint restore does with the offsets in
// its manifest.
func TestSubpageLoadAtRecordedOffsets(t *testing.T) {
	size := validSubpageSizes()[0]
	masks := subpageTestMasks(size)
	tfs := []*subpageTestFile{
		newSubpageTestFile(t, true /* diskBacked */, size, masks, size),
		newSubpageTestFile(t, true /* diskBacked */, size, masks, size),
		newSubpageTestFile(t, true /* diskBacked */, size, masks, 0),
	}
	pages := newPagesFile(t)
	meta, saved := saveAsyncRecorded(t, pages, tfs...)
	for i, s := range saved {
		if s.pagesStart%MinSubpageSize != 0 {
			t.Errorf("file %d starts at pages file offset %#x, which is not a multiple of %d", i, s.pagesStart, MinSubpageSize)
		}
	}
	if hostarch.IsPageAligned(saved[1].pagesStart) {
		t.Fatalf("file 1 starts at page-aligned pages file offset %#x, want an unaligned offset after a sparsely saved file", saved[1].pagesStart)
	}

	apfl := startAsyncLoad(t, pages, subpageLoadConfig{})
	restored := make([]*MemoryFile, len(tfs))
	for i := len(tfs) - 1; i >= 0; i-- {
		restored[i] = newTestMemoryFile(t, true /* diskBacked */)
		t.Cleanup(restored[i].Destroy)
		r := bytes.NewReader(meta.Bytes()[saved[i].metaStart:saved[i].metaEnd])
		opts := LoadOpts{PagesFile: apfl, PagesFileOffset: saved[i].pagesStart}
		if err := restored[i].LoadFrom(context.Background(), r, &opts); err != nil {
			apfl.MemoryFilesDone()
			t.Fatalf("LoadFrom for file %d at pages file offset %#x: %v", i, saved[i].pagesStart, err)
		}
		t.Cleanup(func() { restored[i].DecRef(tfs[i].fr) })
	}
	apfl.MemoryFilesDone()
	for i, tf := range tfs {
		if err := restored[i].AwaitLoadAll(); err != nil {
			t.Fatalf("AwaitLoadAll for file %d: %v", i, err)
		}
		checkSubpageContents(t, restored[i], tf.fr, tf.want, fmt.Sprintf("file %d", i))
	}
}

func TestSubpageLoadMalformedMetadata(t *testing.T) {
	size := validSubpageSizes()[0]
	tf := newSubpageTestFile(t, true /* diskBacked */, size, subpageTestMasks(size), size)
	pages := newPagesFile(t)
	base := parseMetadata(t, saveAsync(t, pages, tf).Bytes())
	full := fullSubpageMask(size)
	for _, tc := range []struct {
		name   string
		mutate func(pb *pgallocpb.MemoryFileMetadataProto)
		lc     subpageLoadConfig
		sync   bool
	}{
		{name: "unmodified"},
		{name: "invalid subpage size", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) { pb.SubpageSize = 3000 }},
		{name: "page-sized subpages", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) { pb.SubpageSize = hostarch.PageSize }},
		{name: "masks without subpage size", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) { pb.SubpageSize = 0 }},
		{name: "missing mask", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) {
			pb.SubpageMasks = pb.SubpageMasks[:len(pb.SubpageMasks)-1]
		}},
		{name: "extra mask", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) { pb.SubpageMasks = append(pb.SubpageMasks, full) }},
		{name: "zero mask", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) { pb.SubpageMasks[0] = 0 }},
		{name: "mask beyond page", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) { pb.SubpageMasks[0] = ^full }},
		{name: "committed ranges out of order", mutate: func(pb *pgallocpb.MemoryFileMetadataProto) {
			slices.Reverse(pb.MemAcct)
		}},
		{name: "no pages file", sync: true},
		{name: "too few ranges per read", lc: subpageLoadConfig{dstRanges: maxSubpageRuns(size) - 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name == "mask beyond page" && full == ^uint64(0) {
				t.Skip("masks of 64 blocks have no bits beyond the page")
			}
			if tc.lc.dstRanges < 0 || (tc.name == "too few ranges per read" && tc.lc.dstRanges == 0) {
				t.Skip("pages have a single run of blocks")
			}
			if tc.name == "committed ranges out of order" && len(base.MemAcct) < 2 {
				t.Skip("a single committed range")
			}
			pb := proto.Clone(base).(*pgallocpb.MemoryFileMetadataProto)
			if tc.mutate != nil {
				tc.mutate(pb)
			}
			meta := marshalMetadata(t, pb)
			f := newTestMemoryFile(t, true /* diskBacked */)
			t.Cleanup(f.Destroy)
			opts := LoadOpts{}
			if !tc.sync {
				opts.PagesFile = startAsyncLoad(t, pages, tc.lc)
			}
			err := f.LoadFrom(context.Background(), meta, &opts)
			if opts.PagesFile != nil {
				opts.PagesFile.MemoryFilesDone()
			}
			if tc.mutate == nil && !tc.sync && tc.lc.dstRanges == 0 {
				if err != nil {
					t.Fatalf("LoadFrom: %v", err)
				}
				t.Cleanup(func() { f.DecRef(tf.fr) })
				if err := f.AwaitLoadAll(); err != nil {
					t.Fatalf("AwaitLoadAll: %v", err)
				}
				checkSubpageContents(t, f, tf.fr, tf.want, "file")
				return
			}
			if err == nil {
				t.Fatalf("LoadFrom succeeded, want error")
			}
			t.Logf("LoadFrom: %v", err)
			if !f.memAcct.IsEmpty() || len(f.chunksLoad()) != 0 {
				t.Errorf("LoadFrom changed the MemoryFile before failing")
			}
		})
	}
}

// BenchmarkSubpageSaveRestore saves and restores a disk-backed MemoryFile with
// async page saving and loading, with sparse (one block per page), fragmented
// (every other block) and dense contents, with and without subpage saving.
// The pages file is in the page cache when it is loaded.
//
// Subpage saving uses 4 KiB blocks, as --save-subpage-blocks=4k does, if host
// pages are larger. Otherwise that flag saves whole pages, so the benchmark
// uses the smallest valid block size instead, which only tests can select,
// and labels it "-testonly".
//
// Memory is reported as Go heap allocations per op. Peak RSS isn't reported:
// it is the high-water mark of the whole process, across benchmarks, and is
// dominated by the test file and its expected contents.
func BenchmarkSubpageSaveRestore(b *testing.B) {
	size := SubpageSize4K()
	label := fmt.Sprintf("subpage=%d", size)
	if size == 0 {
		size = validSubpageSizes()[0]
		label = fmt.Sprintf("subpage=%d-testonly", size)
	}
	const fileBytes = 64 << 20
	for _, c := range []struct {
		name string
		mask uint64
	}{
		{"sparse", 1},
		{"fragmented", subpageAltMask(size)},
		{"dense", fullSubpageMask(size)},
	} {
		for _, subpageSize := range []uint64{0, size} {
			name := "subpage=0"
			if subpageSize != 0 {
				name = label
			}
			b.Run(c.name+"/"+name, func(b *testing.B) {
				masks := make([]uint64, fileBytes/hostarch.PageSize)
				for i := range masks {
					masks[i] = c.mask
				}
				benchmarkSubpageSaveRestore(b, newSubpageTestFile(b, true /* diskBacked */, size, masks, subpageSize))
			})
		}
	}
}

func benchmarkSubpageSaveRestore(b *testing.B, tf *subpageTestFile) {
	pages := newPagesFile(b)
	var (
		saveDur, loadDur   time.Duration
		cpuDur             time.Duration
		metaBytes, pgBytes int
	)
	cpuTime := func() time.Duration {
		var ru unix.Rusage
		if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
			b.Fatalf("Getrusage: %v", err)
		}
		return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
	}
	// Pages written through the file aren't known to be committed until a
	// save scans them. After that, saves without subpage saving skip the
	// scan, as they do for a restored MemoryFile, while subpage saving scans
	// on every save. Save once first so that both are measured in that steady
	// state.
	saveAsync(b, pages, tf)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		if err := pages.Truncate(0); err != nil {
			b.Fatalf("Truncate: %v", err)
		}
		restored := newTestMemoryFile(b, true /* diskBacked */)
		b.StartTimer()

		cpu0 := cpuTime()
		t0 := time.Now()
		meta := saveAsync(b, pages, tf)
		t1 := time.Now()
		metaBytes = meta.Len()
		apfl := startAsyncLoad(b, pages, subpageLoadConfig{})
		if err := restored.LoadFrom(context.Background(), meta, &LoadOpts{PagesFile: apfl}); err != nil {
			b.Fatalf("LoadFrom: %v", err)
		}
		apfl.MemoryFilesDone()
		if err := restored.AwaitLoadAll(); err != nil {
			b.Fatalf("AwaitLoadAll: %v", err)
		}
		t2 := time.Now()
		cpuDur += cpuTime() - cpu0
		saveDur += t1.Sub(t0)
		loadDur += t2.Sub(t1)

		b.StopTimer()
		st, err := pages.Stat()
		if err != nil {
			b.Fatalf("Stat: %v", err)
		}
		pgBytes = int(st.Size())
		restored.DecRef(tf.fr)
		restored.Destroy()
		b.StartTimer()
	}
	b.StopTimer()
	n := float64(b.N)
	b.ReportMetric(float64(saveDur.Microseconds())/n/1e3, "save-ms/op")
	b.ReportMetric(float64(loadDur.Microseconds())/n/1e3, "restore-ms/op")
	b.ReportMetric(float64(cpuDur.Microseconds())/n/1e3, "cpu-ms/op")
	b.ReportMetric(float64(metaBytes), "metadata-B")
	b.ReportMetric(float64(pgBytes), "pagesfile-B")
}
