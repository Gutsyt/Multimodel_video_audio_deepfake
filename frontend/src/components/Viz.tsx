import { animate, useMotionValue } from "motion/react";
import { useEffect, useState } from "react";

const EASE = [0.22, 1, 0.36, 1] as const;

/** Number that counts up to its value. */
export function CountUp({ value, suffix = "%" }: { value: number; suffix?: string }) {
  const mv = useMotionValue(0);
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const c = animate(mv, value, { duration: 1.1, ease: EASE });
    const u = mv.on("change", (v) => setShown(Math.round(v)));
    return () => { c.stop(); u(); };
  }, [mv, value]);
  return <>{shown}{suffix}</>;
}
