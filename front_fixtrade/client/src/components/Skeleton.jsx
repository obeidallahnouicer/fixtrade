export const SkeletonIndexCard = () => (
  <div className="glass-card rounded-3xl p-6 md:p-8 mb-6 animate-pulse">
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div className="flex-1">
        <div className="h-8 md:h-10 bg-gray-300 dark:bg-gray-700 rounded w-48 mb-2"></div>
        <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-64"></div>
      </div>
      <div className="flex-1 text-left md:text-right">
        <div className="h-12 md:h-14 bg-gray-300 dark:bg-gray-700 rounded w-32 mb-2 md:ml-auto"></div>
        <div className="h-6 md:h-7 bg-gray-300 dark:bg-gray-700 rounded w-40 md:ml-auto"></div>
      </div>
    </div>
  </div>
);

export const SkeletonStockCard = () => (
  <div className="glass-card rounded-2xl p-4 md:p-5 min-h-[160px] animate-pulse">
    <div className="flex items-start justify-between mb-3">
      <div className="flex-1">
        <div className="h-5 md:h-6 bg-gray-300 dark:bg-gray-700 rounded w-32 mb-2"></div>
        <div className="h-3 md:h-4 bg-gray-300 dark:bg-gray-700 rounded w-16"></div>
      </div>
      <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-16"></div>
    </div>
    <div className="mb-3">
      <div className="h-8 md:h-10 bg-gray-300 dark:bg-gray-700 rounded w-28 mb-2"></div>
      <div className="h-4 md:h-5 bg-gray-300 dark:bg-gray-700 rounded w-24"></div>
    </div>
    <div className="grid grid-cols-2 gap-2 pt-3 border-t border-gray-200 dark:border-gray-700">
      <div>
        <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-16 mb-1"></div>
        <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-20"></div>
      </div>
      <div>
        <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-16 mb-1"></div>
        <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-20"></div>
      </div>
    </div>
  </div>
);

export const SkeletonDashboard = () => (
  <div className="min-h-screen p-4 md:p-6 lg:p-8">
    <SkeletonIndexCard />
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
      {[...Array(8)].map((_, i) => (
        <SkeletonStockCard key={i} />
      ))}
    </div>
  </div>
);

const Skeleton = {
  IndexCard: SkeletonIndexCard,
  StockCard: SkeletonStockCard,
  Dashboard: SkeletonDashboard,
};

export default Skeleton;
