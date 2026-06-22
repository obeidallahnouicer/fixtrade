import { FormEvent, useEffect, useState } from "react";
import { Activity, ArrowRight, Lock, Mail, Sparkles, UserPlus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { useAuthStore } from "@/store/useAuthStore";

export function AuthPage() {
  const { mode, setMode, signIn, signUp, loading, error, user } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      setPassword("");
    }
  }, [user]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError(null);

    if (mode === "register" && password.length < 8) {
      setFormError("Password must be at least 8 characters long.");
      return;
    }

    if (mode === "login") {
      await signIn(email, password);
      return;
    }

    await signUp(email, password, fullName);
  }

  const accentLabel = mode === "login" ? "Welcome back" : "Create account";

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(16,185,129,0.16),_transparent_30%),linear-gradient(180deg,#09090b_0%,#0f1115_100%)] text-[#ededed] flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-6xl grid lg:grid-cols-[1.1fr_0.9fr] gap-6 items-stretch">
        <section className="rounded-3xl border border-emerald-500/20 bg-black/30 backdrop-blur-xl p-8 md:p-12 overflow-hidden relative">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(16,185,129,0.18),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(59,130,246,0.14),transparent_30%)] pointer-events-none" />
          <div className="relative z-10 flex flex-col h-full gap-8">
            <div className="flex items-center gap-3 text-emerald-400 font-semibold tracking-tight text-xl">
              <Activity fill="currentColor" size={20} />
              FixTrade
            </div>

            <div className="space-y-5 max-w-xl">
              <Badge variant="success" className="w-fit">Microservices thesis mode</Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-none text-white">
                {accentLabel}
              </h1>
              <p className="text-zinc-300 text-base md:text-lg leading-7 max-w-2xl">
                Secure access for market intelligence, predictions, GenAI explanations, and portfolio workflows.
                The frontend now authenticates against the FastAPI backend before opening the dashboard.
              </p>
            </div>

            <div className="grid sm:grid-cols-3 gap-3 max-w-2xl">
              {[
                ["JWT auth", "Secure session layer"],
                ["Role checks", "Foundation for user/admin flows"],
                ["Local Postgres", "Docker-backed persistence"],
              ].map(([title, subtitle]) => (
                <div key={title} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="text-sm font-semibold text-white">{title}</div>
                  <div className="text-xs text-zinc-400 mt-1">{subtitle}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <Card className="bg-[#0f1115] border-zinc-800 shadow-2xl shadow-black/30">
          <CardHeader className="space-y-3 border-b border-zinc-800/80 pb-5">
            <CardTitle className="text-2xl text-white flex items-center gap-2">
              <Sparkles size={20} className="text-emerald-400" />
              {mode === "login" ? "Sign in" : "Create your account"}
            </CardTitle>
            <p className="text-sm text-zinc-400">
              Use the same credentials for the dashboard, recommendations, and future service boundaries.
            </p>
          </CardHeader>

          <CardContent className="pt-6">
            <form className="space-y-4" onSubmit={handleSubmit}>
              {mode === "register" && (
                <div>
                  <label className="text-xs uppercase tracking-[0.2em] text-zinc-500 mb-2 block">Full name</label>
                  <div className="flex items-center gap-3 rounded-2xl border border-zinc-800 bg-black/30 px-4 py-3">
                    <UserPlus size={18} className="text-zinc-500" />
                    <input
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="Your full name"
                      className="w-full bg-transparent outline-none text-sm text-white placeholder:text-zinc-600"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="text-xs uppercase tracking-[0.2em] text-zinc-500 mb-2 block">Email</label>
                <div className="flex items-center gap-3 rounded-2xl border border-zinc-800 bg-black/30 px-4 py-3">
                  <Mail size={18} className="text-zinc-500" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@example.com"
                    className="w-full bg-transparent outline-none text-sm text-white placeholder:text-zinc-600"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs uppercase tracking-[0.2em] text-zinc-500 mb-2 block">Password</label>
                <div className="flex items-center gap-3 rounded-2xl border border-zinc-800 bg-black/30 px-4 py-3">
                  <Lock size={18} className="text-zinc-500" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    minLength={8}
                    placeholder="Minimum 8 characters"
                    className="w-full bg-transparent outline-none text-sm text-white placeholder:text-zinc-600"
                  />
                </div>
                {mode === "register" && (
                  <p className="mt-2 text-xs text-zinc-500">Use at least 8 characters to create an account.</p>
                )}
              </div>

              {(formError || error) && (
                <div className="rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                  {formError || error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-black transition hover:bg-emerald-400 disabled:opacity-60"
              >
                {loading ? "Working..." : mode === "login" ? "Enter dashboard" : "Create account"}
                <ArrowRight size={16} />
              </button>
            </form>

            <div className="mt-6 text-sm text-zinc-400 flex items-center justify-between gap-3 flex-wrap">
              <span>
                {mode === "login" ? "No account yet?" : "Already registered?"}
              </span>
              <button
                type="button"
                onClick={() => setMode(mode === "login" ? "register" : "login")}
                className="text-emerald-400 font-semibold hover:text-emerald-300 transition-colors"
              >
                {mode === "login" ? "Switch to registration" : "Switch to login"}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}