import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    {
      path: '/assignments',
      name: 'assignments-list',
      component: () => import('../views/AssignmentsListView.vue')
    },
    {
      path: '/assignments/new',
      name: 'assignment-create',
      component: () => import('../views/CreateAssignmentView.vue')
    },
    {
      path: '/assignments/:id',
      name: 'assignment-details',
      component: () => import('../views/AssignmentDetailView.vue'),
      props: true
    },
    {
      path: '/grade-exam',
      name: 'grade-exam',
      component: () => import('../views/ExamGradingView.vue')
    }
  ]
})

export default router