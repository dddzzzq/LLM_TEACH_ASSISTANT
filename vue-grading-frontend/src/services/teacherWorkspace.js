// One teacher workspace; existing administrator accounts retain their permissions.
export const isTeacherAccount = (role) => ['teacher', 'admin'].includes(role);
export const mainNavigation = [
  { name: 'AI 助手', path: '/ai-assistant', icon: 'chat' },
  { name: '作业批改', path: '/grade-homework', icon: 'check' },
];
export const homeworkNavigation = [
  { name: '开始批改', path: '/grade-homework' },
  { name: '批改记录', path: '/assignments' },
  { name: '作业材料', path: '/materials' },
  { name: '教务账号', path: '/settings/portal' },
];
export const isHomeworkPage = (path) =>
  [
    '/grade-homework',
    '/assignments',
    '/materials',
    '/settings/portal',
    '/tools-admin',
    '/skills-admin',
    '/logs',
  ].some((prefix) => path === prefix || path.startsWith(prefix + '/'));
