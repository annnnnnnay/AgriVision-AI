import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { Leaf, Mail, Lock, LogIn, Loader2, ArrowLeft } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import GoogleSignInButton from "../components/auth/GoogleSignInButton";

export default function Login() {
  const navigate = useNavigate();
  const { login, signInWithGoogle, resetPassword, isSupabaseConfigured } =
    useAuth();
  const [form, setForm] = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [resetMode, setResetMode] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.email || !form.password) {
      toast.error("Please fill in all fields.");
      return;
    }

    setLoading(true);
    try {
      await login(form.email, form.password);
      toast.success("Welcome back!");
      navigate("/analyze");
    } catch (err) {
      toast.error(err.message || "Login failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setGoogleLoading(true);
    try {
      await signInWithGoogle();
    } catch (err) {
      toast.error(err.message || "Google sign-in failed.");
      setGoogleLoading(false);
    }
  };

  const handlePasswordReset = async (e) => {
    e.preventDefault();
    if (!form.email.trim()) {
      toast.error("Enter your account email first.");
      return;
    }

    setResetLoading(true);
    try {
      await resetPassword(form.email);
      toast.success("If an account exists, a password reset link was sent.");
      setResetMode(false);
    } catch (err) {
      toast.error(err.message || "Could not send the password reset email.");
    } finally {
      setResetLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-brand-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Leaf className="w-8 h-8 text-brand-600" />
          </div>
          <h1 className="font-display text-3xl font-bold text-slate-900">
            Welcome Back
          </h1>
          <p className="mt-2 text-slate-500">
            Sign in to access your crop analysis history
          </p>
        </div>

        <div className="glass-card p-8">
          {resetMode ? (
            <form onSubmit={handlePasswordReset} className="space-y-5">
              <p className="text-sm text-slate-600">
                Enter your email address and we’ll send a secure password reset link.
              </p>
              <label className="block text-sm font-medium text-slate-700">
                Email Address
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  className="input-field mt-1.5"
                  placeholder="farmer@example.com"
                  autoComplete="email"
                  required
                />
              </label>
              <button
                type="submit"
                disabled={resetLoading}
                className="btn-primary w-full py-3.5 text-base"
              >
                {resetLoading ? (
                  <><Loader2 className="h-5 w-5 animate-spin" /> Sending link...</>
                ) : (
                  "Send reset link"
                )}
              </button>
              <button
                type="button"
                onClick={() => setResetMode(false)}
                className="flex w-full items-center justify-center gap-2 text-sm font-medium text-brand-600"
              >
                <ArrowLeft className="h-4 w-4" /> Back to sign in
              </button>
            </form>
          ) : (
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="mb-1.5 block text-sm font-medium text-slate-700">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) =>
                    setForm({ ...form, email: e.target.value })
                  }
                  className="input-field pl-11"
                  placeholder="farmer@example.com"
                  autoComplete="email"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="password"
                  value={form.password}
                  onChange={(e) =>
                    setForm({ ...form, password: e.target.value })
                  }
                  className="input-field pl-11"
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>

            {isSupabaseConfigured && (
              <div className="text-right">
                <button
                  type="button"
                  onClick={() => setResetMode(true)}
                  className="text-sm font-medium text-brand-600 hover:text-brand-700"
                >
                  Forgot password?
                </button>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || googleLoading}
              className="btn-primary w-full py-3.5 text-base"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" /> Signing in...
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" /> Sign In
                </>
              )}
            </button>
          </form>
          )}

          {isSupabaseConfigured && (
            <GoogleSignInButton
              onClick={handleGoogleSignIn}
              disabled={loading || resetLoading}
              loading={googleLoading}
            />
          )}

          <div className="mt-6 text-center">
            <p className="text-sm text-slate-500">
              Don't have an account?{" "}
              <Link
                to="/auth/register"
                className="text-brand-600 font-semibold hover:text-brand-700 transition-colors"
              >
                Create one
              </Link>
            </p>
          </div>
        </div>

        <p className="text-center text-xs text-slate-400 mt-6">
          By signing in, you agree to our Terms of Service and Privacy Policy.
        </p>
      </motion.div>
    </div>
  );
}
