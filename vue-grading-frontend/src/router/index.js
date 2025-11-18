import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import HomeworkGradingView from '../views/HomeworkGradingView.vue'
import AssignmentsListView from '../views/AssignmentsListView.vue'
import AssignmentDetailView from '../views/AssignmentDetailView.vue'
import CreateAssignmentView from '../views/CreateAssignmentView.vue'

// --- 导入新的试卷视图 ---
import ExamListView from '../views/ExamListView.vue'
import CreateExamView from '../views/CreateExamView.vue'
import ExamDetailView from '../views/ExamDetailView.vue'
import StudentReportView from '../views/StudentReportView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      // 修改：根路径重定向到主页
      path: '/',
      redirect: '/home'
    },
    {
      path: '/home',
      name: 'home',
      component: HomeView
    },
    // --- 作业路由 ---
    {
      path: '/grade-homework',
      name: 'grade-homework',
      component: HomeworkGradingView
    },
    {
      path: '/assignments',
      name: 'assignments-list',
      component: AssignmentsListView
    },
    {
      path: '/assignments/new',
      name: 'create-assignment',
      component: CreateAssignmentView
    },
    {
      path: '/assignments/:id',
      name: 'assignment-detail',
      component: AssignmentDetailView,
      props: true
    },
    
    // --- 试卷路由 ---
    {
      // 原 /grade-exam 路径，重定向到试卷列表
      path: '/grade-exam',
      redirect: '/exams'
    },
    {
      path: '/exams',
      name: 'exams-list',
      component: ExamListView
    },
    {
      path: '/exams/new',
      name: 'create-exam',
      component: CreateExamView
    },
    {
      path: '/exams/:id',
      name: 'exam-detail',
      component: ExamDetailView,
      props: true
    },
    {
      path: '/exams/:id/student/:studentExamId', // 使用 studentExamId
      name: 'student-report',
      component: StudentReportView,
      props: true
    }
  ]
})

export default router