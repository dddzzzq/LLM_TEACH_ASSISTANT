// Upgrade a named teacher-owned material batch without grading it.
package main

import (
	"context"
	"flag"
	"fmt"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"grading-gateway/internal/database"
	"grading-gateway/internal/materials"
	"log"
	"os"
	"strings"
)

func main() {
	owner := flag.Uint("owner", 0, "teacher owner ID")
	ids := flag.String("batches", "", "explicit comma-separated batch IDs")
	apply := flag.Bool("apply", false, "update manifests from retained originals")
	flag.Parse()
	if *owner == 0 || strings.TrimSpace(*ids) == "" {
		log.Fatal("owner and batches are required")
	}
	db, err := gorm.Open(mysql.Open(os.Getenv("DATABASE_DSN")), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		log.Fatal("database unavailable")
	}
	database.DB = db
	service := materials.Default()
	for _, id := range strings.Split(*ids, ",") {
		id = strings.TrimSpace(id)
		batch, err := service.Owned(context.Background(), materials.Actor{UserID: *owner, Role: "teacher"}, id)
		if err != nil {
			log.Fatal(err)
		}
		if *apply {
			batch, err = service.EnsureCurrentReader(context.Background(), materials.Actor{UserID: *owner, Role: "teacher"}, id)
		}
		if err != nil {
			log.Fatal(err)
		}
		manifest, err := materials.LoadManifest(batch.ManifestPath, batch.ManifestSHA256)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("batch=%s reader=%d students=%d status=%s grading_job=%s\n", batch.ID, manifest.ReaderVersion, len(manifest.Students), batch.Status, batch.GradingJobID)
	}
}
