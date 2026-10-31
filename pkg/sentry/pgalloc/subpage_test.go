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
	"os"
	"slices"
	"testing"
	"time"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/memutil"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
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

func newSubpageTestMemoryFile(t *testing.T, diskBacked bool) *MemoryFile {
	t.Helper()
	var file *os.File
	if diskBacked {
		var err error
		if file, err = os.CreateTemp(t.TempDir(), "pgalloc-subpage-test"); err != nil {
			t.Fatalf("CreateTemp: %v", err)
		}
	} else {
		fd, err := memutil.CreateMemFD("pgalloc-subpage-test", 0)
		if err != nil {
			t.Fatalf("CreateMemFD: %v", err)
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
		t.Fatalf("NewMemoryFile: %v", err)
	}
	t.Cleanup(f.Destroy)
	return f
}

// subpageTestContents fills pages of fr with data in selected blocks and
// returns the expected contents of fr and the number of non-zero blocks.
func subpageTestContents(t *testing.T, f *MemoryFile, fr memmap.FileRange, block uint64) ([]byte, uint64) {
	t.Helper()
	want := make([]byte, fr.Length())
	set := func(page, blk uint64) {
		b := want[page*hostarch.PageSize+blk*block:][:block]
		for i := range b {
			b[i] = byte(page*31 + blk*7 + uint64(i)%251 + 1)
		}
	}
	blocks := hostarch.PageSize / block
	// Page 0: first block only. Page 1: two runs. Page 2: full. Page 3: zero.
	// Page 4: last block only. Pages 5 and 9: separated by zero pages.
	set(0, 0)
	set(1, 2)
	set(1, 3)
	set(1, blocks-1)
	for b := uint64(0); b < blocks; b++ {
		set(2, b)
	}
	set(4, blocks-1)
	set(5, 1)
	set(9, 0)
	set(9, blocks/2)
	nonzero := uint64(1 + 3 + blocks + 1 + 1 + 2)
	if _, err := unix.Pwrite(f.FD(), want, int64(fr.Start)); err != nil {
		t.Fatalf("Pwrite: %v", err)
	}
	return want, nonzero
}

func TestSubpageSaveRestore(t *testing.T) {
	for _, tc := range []struct {
		name       string
		diskBacked bool
		subpage    bool
	}{
		{name: "disk-backed", diskBacked: true, subpage: true},
		{name: "memfd is saved whole", diskBacked: false, subpage: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			block := uint64(hostarch.PageSize / 8)
			SetSubpageSaveSize(block)
			defer SetSubpageSaveSize(0)

			f := newSubpageTestMemoryFile(t, tc.diskBacked)
			fr, err := f.Allocate(12*hostarch.PageSize, AllocOpts{Mode: AllocateUncommitted})
			if err != nil {
				t.Fatalf("Allocate: %v", err)
			}
			t.Cleanup(func() { f.DecRef(fr) })
			want, nonzero := subpageTestContents(t, f, fr, block)

			// Save asynchronously to a pages file.
			pages, err := os.CreateTemp(t.TempDir(), "pages")
			if err != nil {
				t.Fatalf("CreateTemp: %v", err)
			}
			defer pages.Close()
			wfd, err := unix.Dup(int(pages.Fd()))
			if err != nil {
				t.Fatalf("Dup: %v", err)
			}
			saveDone := make(chan error, 1)
			apfs, err := StartAsyncPagesFileSave(stateio.NewPagesFileFDWriterDefault(int32(wfd)), func(err error) { saveDone <- err })
			if err != nil {
				t.Fatalf("StartAsyncPagesFileSave: %v", err)
			}
			var meta bytes.Buffer
			if err := f.SaveTo(context.Background(), &meta, &SaveOpts{PagesFile: apfs}); err != nil {
				t.Fatalf("SaveTo: %v", err)
			}
			apfs.MemoryFilesDone()
			if err := <-saveDone; err != nil {
				t.Fatalf("async page saving: %v", err)
			}
			wantSize := nonzero * block
			if !tc.subpage {
				wantSize = 6 * hostarch.PageSize // the non-zero pages
			}
			if st, err := pages.Stat(); err != nil || uint64(st.Size()) != wantSize {
				t.Errorf("pages file size = %v (err %v), want %d", st.Size(), err, wantSize)
			}

			// Restore with async page loading, waiting on a page in the
			// middle first as an application fault would.
			restored := newSubpageTestMemoryFile(t, tc.diskBacked)
			rfd, err := unix.Dup(int(pages.Fd()))
			if err != nil {
				t.Fatalf("Dup: %v", err)
			}
			timer := timing.New("test", time.Now())
			defer timer.End()
			timeline := timer.Fork("restore")
			defer timeline.End()
			apfl, err := StartAsyncPagesFileLoad(stateio.NewPagesFileFDReaderDefault(int32(rfd), 1), nil, timeline, AsyncPagesFileLoadOpts{})
			if err != nil {
				t.Fatalf("StartAsyncPagesFileLoad: %v", err)
			}
			if err := restored.LoadFrom(context.Background(), &meta, &LoadOpts{PagesFile: apfl}); err != nil {
				t.Fatalf("LoadFrom: %v", err)
			}
			apfl.MemoryFilesDone()
			mid := memmap.FileRange{fr.Start + 4*hostarch.PageSize, fr.Start + 5*hostarch.PageSize}
			if amfl := restored.asyncPageLoad.Load(); amfl != nil {
				restored.IncRef(mid, 0)
				if err := amfl.awaitLoad(mid); err != nil {
					t.Fatalf("awaitLoad: %v", err)
				}
				restored.DecRef(mid)
			}
			if err := restored.AwaitLoadAll(); err != nil {
				t.Fatalf("AwaitLoadAll: %v", err)
			}
			t.Cleanup(func() { restored.DecRef(fr) })
			got := make([]byte, 0, fr.Length())
			restored.forEachMappingSlice(fr, func(bs []byte) { got = append(got, bs...) })
			if !bytes.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("restored contents differ from saved contents, first at offset %#x (page %d): got %#x, want %#x", i, uint64(i)/hostarch.PageSize, got[i], want[i])
					}
				}
			}
		})
	}
}
