import axios from 'axios';

// API 基础 URL
const API_URL = 'http://127.0.0.1:8000';

// 创建 axios 实例
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 存储 Token 的键名
const ACCESS_TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';
const USER_INFO_KEY = 'user_info';

// 从 localStorage 获取 Token
const getAccessToken = () => localStorage.getItem(ACCESS_TOKEN_KEY);
const getRefreshToken = () => localStorage.getItem(REFRESH_TOKEN_KEY);

// 存储 Token 到 localStorage
const setTokens = (accessToken, refreshToken) => {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
};

// 清除 Token
const clearTokens = () => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_INFO_KEY);
};

// 存储用户信息
const setUserInfo = (userInfo) => {
  localStorage.setItem(USER_INFO_KEY, JSON.stringify(userInfo));
};

// 获取用户信息
const getUserInfo = () => {
  const userInfoStr = localStorage.getItem(USER_INFO_KEY);
  return userInfoStr ? JSON.parse(userInfoStr) : null;
};

// 清除用户信息
const clearUserInfo = () => {
  localStorage.removeItem(USER_INFO_KEY);
};

// 请求拦截器：自动附加 Access Token
apiClient.interceptors.request.use(
  (config) => {
    const token = getAccessToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 刷新 Token 的标记，防止重复刷新
let isRefreshing = false;
let failedQueue = [];

// 处理队列中的失败请求
const processQueue = (error, token = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

// 响应拦截器：处理 401 错误，自动刷新 Token
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    // 如果是 401 错误且不是刷新 Token 的请求
    if (error.response?.status === 401 && !originalRequest._retry && originalRequest.url !== '/api/refresh') {
      if (isRefreshing) {
        // 如果已经在刷新 Token，将请求加入队列
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return apiClient(originalRequest);
          })
          .catch((err) => {
            return Promise.reject(err);
          });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        // 尝试刷新 Token
        const refreshToken = getRefreshToken();
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }

        const response = await axios.post(`${API_URL}/api/refresh`, {
          refresh_token: refreshToken,
        });

        const { access_token, refresh_token, user_id, role, name } = response.data;

        // 存储新的 Token
        setTokens(access_token, refresh_token);

        // 更新用户信息
        const userInfo = getUserInfo();
        if (userInfo) {
          userInfo.user_id = user_id;
          userInfo.role = role;
          userInfo.name = name;
          setUserInfo(userInfo);
        }

        // 更新 Authorization 头
        apiClient.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
        originalRequest.headers.Authorization = `Bearer ${access_token}`;

        // 处理队列中的请求
        processQueue(null, access_token);

        // 重试原始请求
        return apiClient(originalRequest);
      } catch (refreshError) {
        // 刷新 Token 失败，清除本地存储并重定向到登录页
        clearTokens();
        clearUserInfo();
        processQueue(refreshError, null);

        // 重定向到登录页（这里需要根据实际路由配置调整）
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }

        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    // 其他错误直接返回
    return Promise.reject(error);
  }
);

// 导出认证 API 方法
export default {
  // 登录
  login(credentials) {
    return apiClient.post('/api/login', credentials);
  },

  // 注册
  register(userData) {
    return apiClient.post('/api/register', userData);
  },

  // 刷新 Token
  refreshToken(refreshToken) {
    return apiClient.post('/api/refresh', { refresh_token: refreshToken });
  },

  // 获取用户信息
  getProfile() {
    return apiClient.get('/api/profile');
  },

  // 登出
  logout() {
    clearTokens();
    clearUserInfo();
    // 清除 axios 默认 Authorization 头
    delete apiClient.defaults.headers.common['Authorization'];
  },

  // 检查是否已登录
  isAuthenticated() {
    return !!getAccessToken();
  },

  // 获取当前用户信息
  getCurrentUser() {
    return getUserInfo();
  },

  // 获取 Access Token
  getAccessToken,

  // 设置用户信息（登录成功后调用）
  setUserInfo: (userInfo) => {
    setUserInfo(userInfo);
    // 同时存储 Token
    if (userInfo.access_token && userInfo.refresh_token) {
      setTokens(userInfo.access_token, userInfo.refresh_token);
    }
  },

  // 获取 axios 实例（用于其他 API 调用）
  getClient() {
    return apiClient;
  },

  // 根据用户角色获取 API 前缀路径
  getRoleBasedApiPath() {
    const userInfo = this.getCurrentUser();
    if (!userInfo || !userInfo.role) {
      return '/api/student'; // 默认
    }
    const role = userInfo.role;
    if (role === 'teacher') {
      return '/api/teacher';
    } else if (role === 'admin') {
      return '/api/admin';
    } else {
      return '/api/student';
    }
  },

  // 获取角色特定的 API 客户端
  getRoleBasedClient() {
    const prefix = this.getRoleBasedApiPath();
    return {
      get: (url, config) => apiClient.get(prefix + url, config),
      post: (url, data, config) => apiClient.post(prefix + url, data, config),
      put: (url, data, config) => apiClient.put(prefix + url, data, config),
      delete: (url, config) => apiClient.delete(prefix + url, config)
    };
  },

  // 初始化（应用启动时调用）
  init() {
    const token = getAccessToken();
    if (token) {
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
  },
};
