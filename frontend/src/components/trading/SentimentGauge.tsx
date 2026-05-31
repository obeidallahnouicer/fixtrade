import React from "react";
import { cn } from "@/lib/utils";

interface SentimentGaugeProps {
  score: number; // -1 to 1
}

export function SentimentGauge({ score }: SentimentGaugeProps) {
  // Normalize score from [-1, 1] to [0, 100] for percentage
  const percentage = ((score + 1) / 2) * 100;
  
  let label = "Neutral";
  let color = "bg-zinc-500";
  let textColor = "text-zinc-400";
  
  if (score > 0.3) {
    label = "Bullish";
    color = "bg-emerald-500";
    textColor = "text-emerald-400";
  } else if (score < -0.3) {
    label = "Bearish";
    color = "bg-red-500";
    textColor = "text-red-400";
  }

  return (
    <div className="flex flex-col space-y-2">
      <div className="flex justify-between items-end">
        <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wide">
          News Sentiment
        </span>
        <div className="flex items-center gap-2">
          <span className={cn("text-xs font-medium uppercase", textColor)}>
            {label}
          </span>
          <span className="text-sm font-mono text-zinc-100">
            {score > 0 ? "+" : ""}{score.toFixed(2)}
          </span>
        </div>
      </div>
      
      <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden flex">
        {/* Dynamic bar */}
        <div 
          className={cn("h-full transition-all duration-1000 ease-out", color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      <div className="flex justify-between text-[10px] uppercase text-zinc-600 font-mono tracking-widest mt-1">
        <span>Ext Fear (-1)</span>
        <span>Neutral (0)</span>
        <span>Ext Greed (+1)</span>
      </div>
    </div>
  );
}
