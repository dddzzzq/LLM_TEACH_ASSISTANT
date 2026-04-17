import { createRouter, createWebHistory } from 'vue-router'
import authApi from '../services/authApi'

// 导入视图组件
import HomeView from '../views/HomeView.vue'
import HomeworkGradingView from '../views/HomeworkGradingView.vue'
import AssignmentsListView from '../views/AssignmentsListView.vue'
import AssignmentDetailView from '../views/AssignmentDetailView.vue'
import CreateAssignmentView from '../views/CreateAssignmentView.vue'
import ExamListView from '../views/ExamListView.vue'
import CreateExamView from '../views/CreateExamView.vue'
import ExamDetailView from '../views/ExamDetailView.vue'
import StudentReportView from '../views/StudentReportView.vue'
import AIAssistantView from '../views/AIAssistantView.vue'
import LoginView from '../views/LoginView.vue'
import SkillsAdminView from '../views/SkillsAdminView.vue'

// 路由守卫：检查用户是否已认证
const requireAuth = (to, from, next) => {
  if (authApi.isAuthenticated()) {
    next()
  } else {
    next('/login')
  }
}

// 路由守卫：检查角色权限
const checkRolePermission = (allowedRoles) => {
  return (to, from, next) => {
    if (!authApi.isAuthenticated()) {
      next('/login')
      return
    }

    const currentUser = authApi.getCurrentUser()
    if (!currentUser) {
      next('/login')
      return
    }

    const userRole = currentUser.role || 'student'

    // 如果是学生，只能访问AI教学助手
    if (userRole === 'student') {
      if (to.path === '/ai-assistant') {
        next()
      } else {
        next('/ai-assistant')
      }
      return
    }

    // 教师和管理员可以访问所有功能
    if (allowedRoles.includes(userRole)) {
      next()
    } else {
      next('/home')
    }
  }
}

// 路由守卫：如果已登录，重定向到首页
const redirectIfAuthenticated = (to, from, next) => {
  if (authApi.isAuthenticated()) {
    const currentUser = authApi.getCurrentUser()
    if (currentUser) {
      const userRole = currentUser.role || 'student'
      if (userRole === 'student') {
        next('/ai-assistant')
      } else {
        next('/home')
      }
    } else {
      next('/home')
    }
  } else {
    next()
  }
}

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    // 根路径重定向
    {
      path: '/',
      redirect: () => {
        if (!authApi.isAuthenticated()) {
          return '/login'
        }
        const currentUser = authApi.getCurrentUser()
        const userRole = currentUser?.role || 'student'
        if (userRole === 'student') return '/ai-assistant'
        return '/home'
      }
    },
    
    // 公开路由
    {
      path: '/login',
      name: 'login',
      component: LoginView,
      beforeEnter: redirectIfAuthenticated
    },
    
    // 主页路由
    {
      path: '/home',
      name: 'home',
      component: HomeView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 作业批改路由
    {
      path: '/grade-homework',
      name: 'grade-homework',
      component: HomeworkGradingView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 作业列表路由
    {
      path: '/assignments',
      name: 'assignments',
      component: AssignmentsListView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 创建作业路由
    {
      path: '/assignments/new',
      name: 'create-assignment',
      component: CreateAssignmentView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 作业详情路由
    {
      path: '/assignments/:id',
      name: 'assignment-detail',
      component: AssignmentDetailView,
      props: true,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 试卷列表路由
    {
      path: '/exams',
      name: 'exams',
      component: ExamListView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 创建试卷路由
    {
      path: '/exams/new',
      name: 'create-exam',
      component: CreateExamView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 试卷详情路由
    {
      path: '/exams/:id',
      name: 'exam-detail',
      component: ExamDetailView,
      props: true,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 学生报告路由
    {
      path: '/exams/:id/student/:studentExamId',
      name: 'student-report',
      component: StudentReportView,
      props: true,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // AI教学助手路由（所有角色都可以访问）
    {
      path: '/ai-assistant',
      name: 'ai-assistant',
      component: AIAssistantView,
      beforeEnter: requireAuth
    },

    // Skills 管理（教师/管理员）
    {
      path: '/skills-admin',
      name: 'skills-admin',
      component: SkillsAdminView,
      beforeEnter: checkRolePermission(['teacher', 'admin'])
    },
    
    // 404页面
    {
      path: '/:pathMatch(.*)*',
      redirect: '/'
    }
  ]
})

// 全局路由守卫
router.beforeEach((to, from, next) => {
  next()
})

export default router