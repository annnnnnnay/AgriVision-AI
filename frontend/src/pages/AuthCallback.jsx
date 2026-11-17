import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Loader2, Lock, Leaf } from "lucide-react";
import toast from "react-hot-toast";
import { useAuth } from "../context/AuthContext";

export default function AuthCallback() {
  const { isAuthenticated, loading, updatePassword, isSupabaseConfigured } =
    useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [updating, setUpdating] = useState(false);

  const params = new URLSearchParams(location.search);
  const hashParams = new URLSearchParams(location.hash.replace(/^#/, ""));
  const recoveryFlow = params.get("flow") === "recovery";
  const authError =
    params.get("error_description") || hashParams.get("error_description");

  useEffect(() => {
    if (!loading && isAuthenticated && !recoveryFlow) {
      navigate("/analyze", { replace: true });
    }
  }, [isAuthenticated, loading, navigate, recoveryFlow]);

  const handleUpdatePassword = async (event) => {
    event.preventDefault();
    if (password.length < 8) {
      toast.error("Password must be at least 8 characters.");
      return;
    }
    if (password !== confirmation) {
      toast.error("Passwords do not match.");
      return;
    }

    setUpdating(true);
    try {
      await updatePassword(password);
      toast.success("Password updated. You are signed in.");
      navigate("/analyze", { replace: true });
    } catch (error) {
      toast.error(error.message || "Could not update your password.");
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 py-12">
      <div className="glass-card w-full max-w-md p-8 text-center">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-100">
          <Leaf className="h-8 w-8 text-brand-600" />
        </div>
        {loading ? (
          <>
            <Loader2 className="mx-auto h-7 w-7 animate-spin text-brand-600" />
            <h1 className="mt-4 text-xl font-bold text-slate-900">
              Completing secure sign-in...
            </h1>
          </>
        ) : recoveryFlow && isAuthenticated ? (
          <>
            <h1 className="font-display text-2xl font-bold text-slate-900">
              Choose a new password
            </h1>
            <form onSubmit={handleUpdatePassword} className="mt-6 space-y-4 text-left">
              <label className="block text-sm font-medium text-slate-700">
                New password
                <span className="relative mt-1.5 block">
                  <Lock className="absolute left-3.5 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
                  <input
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    className="input-field pl-11"
                    autoComplete="new-password"
                    minLength={8}
                    required
                  />
                </span>
              </label>
              <label className="block text-sm font-medium text-slate-700">
                Confirm new password
                <span className="relative mt-1.5 block">
                  <Lock className="absolute left-3.5 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
                  <input
                    type="password"
                    value={confirmation}
                    onChange={(event) => setConfirmation(event.target.value)}
                    className="input-field pl-11"
                    autoComplete="new-password"
                    minLength={8}
                    required
                  />
                </span>
              </label>
              <button
                type="submit"
                disabled={updating}
                className="btn-primary w-full py-3"
              >
                {updating ? "Updating..." : "Update password"}
              </button>
            </form>
          </>
        ) : (
          <>
            <h1 className="font-display text-2xl font-bold text-slate-900">
              {authError ? "Sign-in link failed" : "Check your email"}
            </h1>
            <p className="mt-3 text-sm text-slate-600">
              {authError
                ? authError
                : recoveryFlow
                  ? "This password reset link is invalid or expired. Request a new one and try again."
                  : isSupabaseConfigured
                    ? "If you confirmed your email, sign in to continue. Otherwise, open the latest confirmation link sent to your inbox."
                    : "Supabase authentication is not configured for this app."}
            </p>
            <Link to="/auth/login" className="btn-primary mt-6 inline-flex px-6 py-3">
              Go to sign in
            </Link>
          </>
        )}
      </div>
    </div>
  );
}
