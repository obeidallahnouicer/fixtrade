import React from "react";
import { Bot, Target, Clock, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { formatPercentage } from "@/lib/utils";
import { AIRecommendation } from "@/types/trading";
import { motion } from "motion/react";

interface AICardProps {
  recommendation: AIRecommendation;
}

export function AICard({ recommendation }: AICardProps) {
  const isBuy = recommendation.action === "BUY";
  const forecastIsPositive = recommendation.predictedReturn > 0.005;
  const forecastIsNegative = recommendation.predictedReturn < -0.005;
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg bg-zinc-900 border border-zinc-800 p-4 flex flex-col space-y-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-zinc-300">
          <Bot size={18} className="text-blue-400" />
          <span className="font-semibold text-sm">Agent Decision</span>
        </div>
        <Badge variant={isBuy ? "success" : recommendation.action === "SELL" ? "danger" : "neutral"}>
          {recommendation.action} SIGNAL
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col space-y-1">
          <span className="text-xs text-zinc-500 uppercase tracking-wide flex items-center gap-1.5">
            <Target size={12} /> Confidence
          </span>
          <span className="font-mono text-xl text-zinc-100">
            {formatPercentage(recommendation.confidence * 100)}
          </span>
        </div>
        
        <div className="flex flex-col space-y-1">
          <span className="text-xs text-zinc-500 uppercase tracking-wide flex items-center gap-1.5">
            <Clock size={12} /> {recommendation.horizonDays}-Day Forecast
          </span>
          <span className={`font-mono text-xl flex items-center ${
            forecastIsPositive
              ? "text-emerald-400"
              : forecastIsNegative
                ? "text-red-400"
                : "text-zinc-300"
          }`}>
            {forecastIsPositive ? (
              <ArrowUpRight size={18} className="mr-1" />
            ) : forecastIsNegative ? (
              <ArrowDownRight size={18} className="mr-1" />
            ) : (
              <Minus size={18} className="mr-1" />
            )}
            {formatPercentage(recommendation.predictedReturn)}
          </span>
        </div>
      </div>

      <div className="h-px w-full bg-zinc-800" />

      <div className="flex flex-col space-y-2">
        <span className="text-xs font-semibold text-zinc-400">REASONING ENGINE</span>
        <p className="text-sm text-zinc-400 leading-relaxed">
          {recommendation.reasoning}
        </p>
      </div>
    </motion.div>
  );
}
