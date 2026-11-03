// Media the user analysed in THIS tab. Never uploaded again or persisted: lets the result page
// replay the real file under the evidence. Gone on reload (the server deletes its copy).
const media = new Map<string, { url: string; file: File }>();

export function rememberMedia(id: string, file: File): void {
  media.set(id, { url: URL.createObjectURL(file), file });
  for (const [k, v] of [...media].slice(0, -5)) { URL.revokeObjectURL(v.url); media.delete(k); }
}
export const sessionMedia = (id: string) => media.get(id) ?? null;
