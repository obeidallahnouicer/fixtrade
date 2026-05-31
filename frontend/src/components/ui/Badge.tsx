import { cn } from "@/lib/utils";
import React from "react";

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "success" | "danger" | "warning" | "neutral";
  children?: React.ReactNode;
  className?: string;
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variants = {
    default: "bg-zinc-800 text-zinc-100",
    success: "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20",
    danger: "bg-red-500/15 text-red-400 border border-red-500/20",
    warning: "bg-amber-500/15 text-amber-500 border border-amber-500/20",
    neutral: "bg-zinc-800 text-zinc-400 border border-zinc-700",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold uppercase tracking-wider transition-colors",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
