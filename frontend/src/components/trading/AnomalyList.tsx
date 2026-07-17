import React from "react";
import { useStore } from "@/store/useStore";
import { AlertCircle, TrendingUp, TrendingDown, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

export function AnomalyList() {
  const { anomalies } = useStore();

  if (anomalies.length === 0) {
    return (
      <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-4 py-3 text-sm text-zinc-500">
        No anomaly alerts were returned for this symbol yet. The live detector is
        running against the real dataset, but this market slice currently has no alerts.
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-3">
      {anomalies.map((anomaly) => {
        const isCritical = anomaly.severity > 0.8;
        
        let Icon = Activity;
        if (anomaly.type.includes("Spike")) Icon = TrendingUp;
        if (anomaly.type.includes("Contradiction")) Icon = AlertCircle;

        return (
          <div 
            key={anomaly.id} 
            className="flex items-start gap-3 p-3 rounded-md bg-zinc-900 border border-zinc-800/50"
          >
            <div className={cn(
              "p-2 rounded-full mt-0.5 flex-shrink-0",
              isCritical ? "bg-red-500/10 text-red-400" : "bg-amber-500/10 text-amber-500"
            )}>
              <Icon size={16} />
            </div>
            
            <div className="flex flex-col space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wide">
                  {anomaly.symbol} • {anomaly.type}
                </span>
                <span className={cn(
                  "text-[10px] font-mono px-1.5 py-0.5 rounded",
                  isCritical ? "bg-red-500/20 text-red-300" : "bg-amber-500/20 text-amber-300"
                )}>
                  {anomaly.severity.toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-zinc-500 leading-snug">
                {anomaly.description}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
