import { AppDataProvider, useAppData } from "./AppData";
import { Shell } from "./components/Shell";
import { ToastProvider } from "./components/Toasts";
import { useRoute } from "./lib/route";
import { AnalysisPage } from "./pages/AnalysisPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { HistoryPage } from "./pages/HistoryPage";
import { ModelsPage } from "./pages/ModelsPage";

function Routes() {
  const route = useRoute();
  const { caps, capsError } = useAppData();
  return (
    <Shell route={route} caps={caps} capsError={capsError}>
      {route.name === "analyze" && <AnalyzePage />}
      {route.name === "analysis" && <AnalysisPage id={route.id} />}
      {route.name === "history" && <HistoryPage />}
      {route.name === "models" && <ModelsPage />}
    </Shell>
  );
}

export default function App() {
  return <ToastProvider><AppDataProvider><Routes /></AppDataProvider></ToastProvider>;
}
