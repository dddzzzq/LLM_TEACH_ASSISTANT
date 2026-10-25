import { createRouter, createWebHistory } from 'vue-router';
import authApi from '../services/authApi';
import { isTeacherAccount } from '../services/teacherWorkspace';
import HomeworkGradingView from '../views/HomeworkGradingView.vue';
import AssignmentsListView from '../views/AssignmentsListView.vue';
import AssignmentDetailView from '../views/AssignmentDetailView.vue';
import CreateAssignmentView from '../views/CreateAssignmentView.vue';
import AIAssistantView from '../views/AIAssistantView.vue';
import LoginView from '../views/LoginView.vue';
import ToolsAdminView from '../views/ToolsAdminView.vue';
import LogsView from '../views/LogsView.vue';
import MaterialsView from '../views/MaterialsView.vue';
import PortalSettingsView from '../views/PortalSettingsView.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/assignments' },
    { path: '/home', redirect: '/assignments' },
    { path: '/login', name: 'login', component: LoginView, meta: { public: true } },
    { path: '/ai-assistant', name: 'ai-assistant', component: AIAssistantView },
    { path: '/grade-homework', name: 'grade-homework', component: HomeworkGradingView },
    { path: '/assignments', name: 'assignments', component: AssignmentsListView },
    { path: '/assignments/new', name: 'create-assignment', component: CreateAssignmentView },
    {
      path: '/assignments/:id',
      name: 'assignment-detail',
      component: AssignmentDetailView,
      props: true,
    },
    { path: '/materials', name: 'materials', component: MaterialsView },
    { path: '/settings/portal', name: 'portal-settings', component: PortalSettingsView },
    {
      path: '/tools-admin',
      alias: '/skills-admin',
      name: 'skills-admin',
      component: ToolsAdminView,
    },
    { path: '/logs', name: 'logs', component: LogsView, meta: { admin: true } },
    { path: '/:pathMatch(.*)*', redirect: '/assignments' },
  ],
});
router.beforeEach((to) => {
  const signedIn = authApi.isAuthenticated();
  const role = authApi.getCurrentUser()?.role;
  if (signedIn && !isTeacherAccount(role)) {
    authApi.logout();
    return { path: '/login', query: { reason: 'teacher' } };
  }
  if (to.meta.public) return signedIn ? '/assignments' : true;
  if (!signedIn) return '/login';
  if (to.meta.admin && role !== 'admin') return '/assignments';
  return true;
});
export default router;
