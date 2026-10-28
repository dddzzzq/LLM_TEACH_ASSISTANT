// One teacher workspace; existing administrator accounts retain their permissions.
export const isTeacherAccount = (role) => ['teacher', 'admin'].includes(role);
export const mainNavigation = [{ name: '作业工作台', path: '/assignments', icon: 'book' }];
export const isHomeworkPage = (path) =>
  ['/grade-homework', '/assignments', '/materials', '/settings/portal', '/logs'].some(
    (prefix) => path === prefix || path.startsWith(prefix + '/')
  );
