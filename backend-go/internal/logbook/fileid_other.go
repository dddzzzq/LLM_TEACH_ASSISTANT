//go:build !linux

package logbook

import "os"

func fileIdentity(info os.FileInfo) string { return info.Name() }
