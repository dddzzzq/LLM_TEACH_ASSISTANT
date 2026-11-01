package portal

import (
	"context"
	"encoding/json"
	"time"

	"grading-gateway/internal/agent/modelclient"
	"grading-gateway/internal/models"
	"grading-gateway/internal/rpa"
)

func catalogHasDetails(catalog *Catalog) bool {
	if catalog != nil {
		for _, course := range catalog.Courses {
			for _, work := range course.Homeworks {
				if work.Detail != nil {
					return true
				}
			}
		}
	}
	return false
}

func catalogDetailTarget(connectionID, snapshotID string, course CatalogCourse, work CatalogHomework) detailTarget {
	return detailTarget{
		HomeworkID: stableID(connectionID, "homework", course.PlatformID+":"+work.PlatformID),
		SnapshotID: snapshotID, CourseName: course.Title, HomeworkName: work.Title, Term: course.Term,
		Target: rpa.PortalTarget{CourseID: course.PlatformID, HomeworkID: work.PlatformID},
	}
}

// Use the sync job's resource directory and existing parsers. No new browser or
// HOMEWORK_DETAIL operation is needed; a restart can replay the worker receipt.
func resolveCatalogDetails(ctx context.Context, op models.PortalOperation, catalog *Catalog, call requirementModel) (*Catalog, error) {
	if err := validateCatalog(*catalog); err != nil {
		return nil, err
	}
	result := *catalog
	result.Courses = append([]CatalogCourse{}, catalog.Courses...)
	for i := range result.Courses {
		course := &result.Courses[i]
		course.Homeworks = append([]CatalogHomework{}, course.Homeworks...)
		for j := range course.Homeworks {
			work := &course.Homeworks[j]
			if work.Detail == nil {
				continue // Older workers can still use the separate detail queue.
			}
			selected := catalogDetailTarget(op.ConnectionID, "", *course, *work)
			payload, _ := json.Marshal(selected)
			read := op
			read.PayloadJSON = string(payload)
			parseCtx, cancel := context.WithTimeout(ctx, 4*time.Minute)
			detail, err := resolveDetailWithModel(parseCtx, read, work.Detail, call)
			cancel()
			if err != nil {
				return nil, err
			}
			work.Detail = detail
		}
	}
	return &result, nil
}

func (r *Runner) parseCatalogDetails(ctx context.Context, op models.PortalOperation, state operationState) {
	parseCtx, cancel := context.WithCancel(ctx)
	if _, loaded := r.detailParsers.LoadOrStore(op.ID, cancel); loaded {
		cancel()
		return
	}
	go func() {
		defer r.detailParsers.Delete(op.ID)
		defer cancel()
		result, err := resolveCatalogDetails(parseCtx, op, state.Catalog, modelclient.NewDeepSeekClient("").CallJSONMessages)
		if err != nil {
			state.Status, state.Stage, state.ErrorCode, state.Message = "FAILED", "READING_DETAIL", "DETAIL_REJECTED", "同步中的作业详情核验或解析失败，请重新同步"
		} else {
			state.Catalog, state.Message = result, "课程、作业及作业要求已同步，未完整整理的要求可单独重新同步"
		}
		if ctx.Err() != nil {
			return
		}
		if err := r.apply(ctx, op, state); err != nil {
			state.Status, state.ErrorCode, state.Message = "FAILED", "RESULT_REJECTED", "同步结果核验未通过，已保留上次成功目录"
			_ = r.apply(ctx, op, state)
		}
	}()
}
