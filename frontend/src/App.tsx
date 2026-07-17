/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from "react";
import { Dashboard } from "@/pages/Dashboard";
import { AuthPage } from "@/pages/AuthPage";
import { useAuthStore } from "@/store/useAuthStore";

export default function App() {
  const { user, isHydrated, hydrate } = useAuthStore();

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  if (!isHydrated) {
    return (
      <div className="min-h-screen bg-[#09090b] text-zinc-300 flex items-center justify-center">
        Loading FixTrade...
      </div>
    );
  }

  return user ? <Dashboard /> : <AuthPage />;
}
