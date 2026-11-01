package browserworker

import (
	"golang.org/x/sys/windows"
	"os"
)

func lockFile(path string) (*os.File, error) {
	f, e := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if e != nil {
		return nil, e
	}
	e = windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &windows.Overlapped{})
	if e != nil {
		f.Close()
		return nil, e
	}
	return f, nil
}
func unlockFile(f *os.File) {
	if f != nil {
		_ = windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, &windows.Overlapped{})
		_ = f.Close()
	}
}
