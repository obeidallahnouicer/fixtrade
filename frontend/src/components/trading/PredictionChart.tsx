import React, { useMemo } from 'react';
import { 
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { PricePoint } from '@/types/trading';

interface PredictionChartProps {
  data: PricePoint[];
}

export function PredictionChart({ data }: PredictionChartProps) {
  const splitPoint = useMemo(() => {
    const firstForecast = data.find((d) => d.predictedPrice !== undefined);
    return firstForecast?.date;
  }, [data]);

  return (
    <div className="h-[400px] w-full mt-4 font-mono text-xs">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 20, right: 0, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.01} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
          <XAxis 
            dataKey="date" 
            stroke="#52525b" 
            tick={{ fill: '#a1a1aa' }} 
            tickMargin={10} 
            minTickGap={50}
            axisLine={false}
            tickLine={false}
          />
          <YAxis 
            stroke="#52525b" 
            domain={['auto', 'auto']} 
            tick={{ fill: '#a1a1aa' }}
            tickFormatter={(val) => val.toFixed(1)}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '4px', padding: '12px' }}
            itemStyle={{ color: '#f4f4f5', fontFamily: 'JetBrains Mono', fontSize: '12px' }}
            labelStyle={{ color: '#a1a1aa', marginBottom: '8px', fontFamily: 'Inter', fontSize: '12px' }}
            cursor={{ stroke: '#3f3f46', strokeWidth: 1 }}
          />
          
          {/* Confidence interval area */}
          <Area 
            type="monotone" 
            dataKey="confUpper" 
            stroke="none" 
            fill="url(#colorConf)" 
            isAnimationActive={false}
            activeDot={false}
          />
          <Area 
            type="monotone" 
            dataKey="confLower" 
            stroke="none" 
            fill="#121214" 
            isAnimationActive={false}
            activeDot={false}
          />

          {/* Historical line */}
          <Line 
            type="monotone" 
            dataKey="historicalPrice" 
            stroke="#f4f4f5" 
            strokeWidth={2}
            dot={false} 
            isAnimationActive={false}
            name="Actual"
          />

          {/* Predicted line */}
          <Line 
            type="monotone" 
            dataKey="predictedPrice" 
            stroke="#10b981" 
            strokeWidth={2} 
            strokeDasharray="4 4"
            dot={false}
            name="Forecast"
          />
          
          {splitPoint && (
            <ReferenceLine 
              x={splitPoint} 
              stroke="#52525b" 
              strokeDasharray="3 3" 
              label={{ position: 'top', value: 'LIVE', fill: '#a1a1aa', fontSize: 10 }} 
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
