// One teacher workspace; existing administrator accounts retain their permissions.
export const isTeacherAccount = (role) => ['teacher', 'admin'].includes(role);
export const mainNavigation = [
  { name: '作业批改', path: '/assignments', icon: 'check' },
  { name: 'AI 助手', path: '/ai-assistant', icon: 'chat' },
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
