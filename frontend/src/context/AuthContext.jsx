import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import api from "../services/api";
import { supabase } from "../services/supabase";

const AuthContext = createContext(null);

function toProfile(user) {
  if (!user) return null;

  const metadata = user.user_metadata || {};
  return {
    id: user.id,
    name:
      metadata.full_name ||
      metadata.name ||
      metadata.user_name ||
      user.email?.split("@")[0] ||
      "Farmer",
    email: user.email || "",
    location: metadata.location || null,
    created_at: user.created_at || null,
    total_predictions: 0,
  };
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() =>
    supabase ? null : localStorage.getItem("token")
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    if (supabase) {
      localStorage.removeItem("token");
      const { data } = supabase.auth.onAuthStateChange((_event, session) => {
        if (!active) return;
        setToken(session?.access_token || null);
        setUser(toProfile(session?.user));
        setLoading(false);
      });

      supabase.auth
        .getSession()
        .then(({ data: sessionData, error }) => {
          if (!active) return;
          if (error) {
            console.error("Failed to restore Supabase session:", error);
            setToken(null);
            setUser(null);
          } else {
            setToken(sessionData.session?.access_token || null);
            setUser(toProfile(sessionData.session?.user));
          }
          setLoading(false);
        })
        .catch((error) => {
          if (!active) return;
          console.error("Failed to restore Supabase session:", error);
          setToken(null);
          setUser(null);
          setLoading(false);
        });

      return () => {
        active = false;
        data.subscription.unsubscribe();
      };
    }

    const storedToken = localStorage.getItem("token");
    if (!storedToken) {
      setLoading(false);
      return () => {
        active = false;
      };
    }

    api
      .get("/api/v1/auth/me")
      .then((response) => {
        if (active) setUser(response.data.data);
      })
      .catch((error) => {
        if (active) {
          console.error("Failed to restore legacy session:", error);
          localStorage.removeItem("token");
          setToken(null);
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const login = useCallback(async (email, password) => {
    if (supabase) {
      const { data, error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });
      if (error) throw error;
      if (!data.session || !data.user) {
        throw new Error("Email confirmation is required before signing in.");
      }
      return toProfile(data.user);
    }

    const response = await api.post("/api/v1/auth/login", {
      email: email.trim(),
      password,
    });
    const { user: userData, token: authToken } = response.data.data;
    setUser(userData);
    setToken(authToken);
    localStorage.setItem("token", authToken);
    return userData;
  }, []);

  const register = useCallback(async (name, email, password, location) => {
    if (supabase) {
      const { data, error } = await supabase.auth.signUp({
        email: email.trim(),
        password,
        options: {
          data: {
            full_name: name.trim(),
            location: location.trim() || null,
          },
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      });
      if (error) throw error;

      if (data.session && data.user) {
        return { user: toProfile(data.user), requiresEmailConfirmation: false };
      }
      return {
        user: toProfile(data.user),
        requiresEmailConfirmation: true,
      };
    }

    const response = await api.post("/api/v1/auth/register", {
      name: name.trim(),
      email: email.trim(),
      password,
      location: location.trim() || null,
    });
    const { user: userData, token: authToken } = response.data.data;
    setUser(userData);
    setToken(authToken);
    localStorage.setItem("token", authToken);
    return { user: userData, requiresEmailConfirmation: false };
  }, []);

  const signInWithGoogle = useCallback(async () => {
    if (!supabase) {
      throw new Error(
        "Google sign-in is unavailable because Supabase is not configured in the frontend."
      );
    }

    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
        queryParams: { prompt: "select_account" },
      },
    });
    if (error) throw error;
  }, []);

  const resetPassword = useCallback(async (email) => {
    if (!supabase) {
      throw new Error("Password reset requires Supabase Auth configuration.");
    }

    const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/auth/callback?flow=recovery`,
    });
    if (error) throw error;
  }, []);

  const updatePassword = useCallback(async (password) => {
    if (!supabase) {
      throw new Error("Password updates require Supabase Auth configuration.");
    }

    const { data, error } = await supabase.auth.updateUser({ password });
    if (error) throw error;
    return toProfile(data.user);
  }, []);

  const logout = useCallback(async () => {
    if (supabase) {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
    } else {
      localStorage.removeItem("token");
      setUser(null);
      setToken(null);
    }
  }, []);

  const value = useMemo(
    () => ({
      user,
      token,
      loading,
      isAuthenticated: !!user,
      isGuestPreview:
        (!supabase && import.meta.env.DEV) ||
        (!user && import.meta.env.VITE_ENABLE_GUEST_DEMO === "true"),
      isSupabaseConfigured: !!supabase,
      login,
      register,
      signInWithGoogle,
      resetPassword,
      updatePassword,
      logout,
    }),
    [
      user,
      token,
      loading,
      login,
      register,
      signInWithGoogle,
      resetPassword,
      updatePassword,
      logout,
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

export default AuthContext;
