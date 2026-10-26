import { AppDataProvider, useAppData } from "./AppData";
import { Shell } from "./components/Shell";
import { ToastProvider } from "./components/Toasts";
import { useRoute } from "./lib/route";
import { AnalysisPage } from "./pages/AnalysisPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { DashboardPage } from "./pages/DashboardPage";
import { HistoryPage } from "./pages/HistoryPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SystemPage } from "./pages/SystemPage";

function Routes() {
  const route = useRoute();
  const { caps, capsError } = useAppData();
  return (
    <Shell route={route} caps={caps} capsError={capsError}>
      {route.name === "dashboard" && <DashboardPage />}
      {route.name === "analyze" && <AnalyzePage />}
      {route.name === "analysis" && <AnalysisPage key={route.id} id={route.id} />}
      {route.name === "history" && <HistoryPage />}
      {route.name === "reports" && <ReportsPage />}
      {route.name === "system" && <SystemPage />}
    </Shell>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <AppDataProvider><Routes /></AppDataProvider>
    </ToastProvider>
  );
}
