import { create } from "zustand";
import { fetchCurrentUser, loginUser, registerUser, AuthUser } from "@/services/api";

const TOKEN_KEY = "fixtrade_auth_token";
const DEMO_KEY = "fixtrade_demo_mode";

type AuthMode = "login" | "register";

interface AuthState {
  user: AuthUser | null;
  token: string | null;
  isHydrated: boolean;
  loading: boolean;
  error: string | null;
  mode: AuthMode;
  setMode: (mode: AuthMode) => void;
  hydrate: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, fullName?: string) => Promise<void>;
  signOut: () => void;
  continueAsGuest: () => void;
}

function saveSession(token: string) {
  localStorage.removeItem(DEMO_KEY);
  localStorage.setItem(TOKEN_KEY, token);
}

function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(DEMO_KEY);
}

function readSession() {
  return localStorage.getItem(TOKEN_KEY);
}

function demoUser(): AuthUser {
  return {
    id: "demo",
    email: "demo@fixtrade.local",
    full_name: "Demo Guest",
    role: "demo",
    is_active: true,
    created_at: new Date().toISOString(),
  };
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  token: null,
  isHydrated: false,
  loading: false,
  error: null,
  mode: "login",
  setMode: (mode) => set({ mode, error: null }),
  hydrate: async () => {
    const token = readSession();
    if (!token) {
      if (localStorage.getItem(DEMO_KEY) === "true") {
        set({ isHydrated: true, token: null, user: demoUser() });
        return;
      }
      set({ isHydrated: true, token: null, user: null });
      return;
    }

    set({ loading: true, error: null, token });
    try {
      const user = await fetchCurrentUser();
      set({ user, isHydrated: true, loading: false });
    } catch {
      clearSession();
      set({ user: null, token: null, isHydrated: true, loading: false });
    }
  },
  signIn: async (email, password) => {
    set({ loading: true, error: null });
    try {
      const response = await loginUser({ email, password });
      saveSession(response.access_token);
      set({ user: response.user, token: response.access_token, loading: false, mode: "login" });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Login failed", loading: false });
    }
  },
  signUp: async (email, password, fullName) => {
    set({ loading: true, error: null });
    try {
      const response = await registerUser({ email, password, full_name: fullName || undefined });
      saveSession(response.access_token);
      set({ user: response.user, token: response.access_token, loading: false, mode: "login" });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Registration failed", loading: false });
    }
  },
  signOut: () => {
    clearSession();
    set({ user: null, token: null, error: null });
  },
  continueAsGuest: () => {
    clearSession();
    localStorage.setItem(DEMO_KEY, "true");
    set({
      user: demoUser(),
      token: null,
      error: null,
      isHydrated: true,
    });
  },
}));
