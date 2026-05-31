import React from "react";
import { useStore } from "@/store/useStore";
import { cn, formatCurrency, formatPercentage } from "@/lib/utils";
import { motion } from "motion/react";

export function StockList() {
  const { stocks, selectedStockId, setSelectedStockId } = useStore();

  return (
    <div className="flex flex-col space-y-1">
      <div className="px-3 pb-2 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
        Watchlist
      </div>
      {stocks.map((stock) => {
        const isSelected = selectedStockId === stock.symbol;
        const isPositive = stock.change >= 0;

        return (
          <motion.div
            key={stock.symbol}
            whileHover={{ x: 2 }}
            onClick={() => setSelectedStockId(stock.symbol)}
            className={cn(
              "group cursor-pointer rounded-md p-3 flex justify-between items-center transition-colors",
              isSelected ? "bg-zinc-800/50" : "hover:bg-zinc-800/30"
            )}
          >
            <div className="flex flex-col">
              <span className={cn("font-medium", isSelected ? "text-zinc-100" : "text-zinc-300")}>
                {stock.symbol}
              </span>
              <span className="text-xs text-zinc-500 truncate max-w-[120px]">
                {stock.name}
              </span>
            </div>
            
            <div className="flex flex-col items-end font-mono">
              <span className="text-zinc-100 text-sm">
                {formatCurrency(stock.price)}
              </span>
              <span className={cn("text-xs", isPositive ? "text-emerald-400" : "text-red-400")}>
                {isPositive ? "+" : ""}{formatPercentage(stock.changePercent)}
              </span>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
