import { Loader2 } from "lucide-react";

export default function GoogleSignInButton({
  onClick,
  disabled = false,
  loading = false,
}) {
  return (
    <>
      <div className="my-5 flex items-center gap-3 text-xs text-slate-400">
        <span className="h-px flex-1 bg-slate-200" />
        or continue with
        <span className="h-px flex-1 bg-slate-200" />
      </div>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled || loading}
        className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-200 bg-white py-3 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : (
          <span
            aria-hidden="true"
            className="flex h-5 w-5 items-center justify-center rounded-full border border-slate-200 font-bold text-blue-600"
          >
            G
          </span>
        )}
        {loading ? "Connecting to Google..." : "Continue with Google"}
      </button>
    </>
  );
}
