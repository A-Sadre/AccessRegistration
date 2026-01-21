import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { PersonalInfo } from "../types";
import { apiUrl } from "../api";

type Props = {
  info: PersonalInfo;
  onBack: () => void; // رجوع لتعديل البيانات (Step 5)
  onBackToMrz: () => void; // رجوع للـ MRZ (Step 3)
  onDone: () => void; // إنهاء
};

type Step = "camera" | "processing" | "result";

type UiResult = {
  kind: "success" | "warn" | "error";
  title: string;
  desc: string;
  action: "back_mrz" | "done";
};

export function FaceEnroll({ info, onBack, onBackToMrz, onDone }: Props) {
  // -----------------------------
  // Config
  // -----------------------------
  const GOOD_FRAMES_REQUIRED = 7;
  const CAPTURE_EVERY_MS = 170;
  const COOLDOWN_MS = 1500;

  const MIN_FACE_AREA = 0.1;
  const MAX_MOVE_PER_FRAME = 0.028;

  const CANDIDATE_BUFFER_MAX = 24;
  const TEMPLATES_TO_SEND = 3;

  const personId = (info?.documentNumber || "").trim();

  // -----------------------------
  // Refs
  // -----------------------------
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const inFlightRef = useRef(false);
  const lastSentAtRef = useRef(0);

  const lastCaptureAtRef = useRef(0);
  const prevBoxRef = useRef<any>(null);
  const selectedBoxRef = useRef<any>(null);

  const candidatesRef = useRef<{ blob: Blob; blurScore: number }[]>([]);
  const framesRef = useRef(0);

  // -----------------------------
  // State
  // -----------------------------
  const [step, setStep] = useState<Step>("camera");
  const [status, setStatus] = useState("ضع وجهك داخل الإطار");
  const [busy, setBusy] = useState(false);
  const [uiResult, setUiResult] = useState<UiResult | null>(null);

  function stopLoop() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }

  function stopCamera() {
    stopLoop();
    const stream = streamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }

  function resetRuntime() {
    inFlightRef.current = false;
    lastSentAtRef.current = 0;
    lastCaptureAtRef.current = 0;
    prevBoxRef.current = null;
    selectedBoxRef.current = null;
    candidatesRef.current = [];
    framesRef.current = 0;

    setBusy(false);
    setUiResult(null);
    setStatus("ضع وجهك داخل الإطار");
  }

  function landmarksToBBox(lm: any[]) {
    let minX = 1,
      minY = 1,
      maxX = 0,
      maxY = 0;
    for (const p of lm) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    const w = maxX - minX;
    const h = maxY - minY;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    return { minX, minY, maxX, maxY, w, h, cx, cy };
  }

  function isStable(box: any) {
    const prev = prevBoxRef.current;
    prevBoxRef.current = box;
    if (!prev) return true;
    const dx = box.cx - prev.cx;
    const dy = box.cy - prev.cy;
    const move = Math.sqrt(dx * dx + dy * dy);
    return move <= MAX_MOVE_PER_FRAME;
  }

  function estimateYaw(lm: any[]) {
    const L = lm[33],
      R = lm[263],
      N = lm[1];
    const eyeDist = Math.abs(R.x - L.x) + 1e-6;
    const noseOffset = Math.abs(N.x - (L.x + R.x) / 2);
    return noseOffset / eyeDist;
  }

  function gateSelectFaceInsideOval(detection: any) {
    const faces = detection?.faceLandmarks || [];
    if (!faces.length) {
      selectedBoxRef.current = null;
      return { ok: false, msg: "ضع وجهك داخل الإطار" };
    }

    const ovalCx = 0.5;
    const ovalCy = 0.52;
    const rx = 0.36;
    const ry = 0.4;

    function insideOval(x: number, y: number) {
      const nx = (x - ovalCx) / rx;
      const ny = (y - ovalCy) / ry;
      return nx * nx + ny * ny <= 1.2;
    }

    const inside = faces
      .map((lm: any[]) => {
        const box = landmarksToBBox(lm);
        const corners = [
          [box.minX, box.minY],
          [box.maxX, box.minY],
          [box.minX, box.maxY],
          [box.maxX, box.maxY],
        ];
        const insideCount = corners.reduce(
          (acc, [x, y]) => acc + (insideOval(x as number, y as number) ? 1 : 0),
          0,
        );
        const faceArea = box.w * box.h;
        const inOval = insideCount >= 3;
        return { lm, box, inOval, insideCount, faceArea };
      })
      .filter((x: any) => x.inOval);

    if (inside.length === 0) {
      selectedBoxRef.current = null;
      return { ok: false, msg: "ضع وجهك داخل الإطار" };
    }

    if (inside.length > 1) {
      selectedBoxRef.current = null;
      return { ok: false, msg: "يرجى وجود شخص واحد فقط" };
    }

    const chosen = inside[0];
    const yaw = estimateYaw(chosen.lm);
    if (yaw > 0.22) {
      selectedBoxRef.current = null;
      return { ok: false, msg: "وجهك للأمام" };
    }

    if (chosen.faceArea < MIN_FACE_AREA) {
      selectedBoxRef.current = null;
      return { ok: false, msg: "اقترب من الكاميرا" };
    }

    selectedBoxRef.current = chosen.box;
    return { ok: true, msg: "جاري الالتقاط…" };
  }

  function computeBlurScoreFromCanvas(capCanvas: HTMLCanvasElement) {
    const tmp = document.createElement("canvas");
    const w = 240,
      h = 240;
    tmp.width = w;
    tmp.height = h;
    const tctx = tmp.getContext("2d")!;
    tctx.drawImage(capCanvas, 0, 0, w, h);

    const img = tctx.getImageData(0, 0, w, h).data;
    const gray = new Float32Array(w * h);
    for (let i = 0, p = 0; i < img.length; i += 4, p++) {
      gray[p] = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
    }

    let sum = 0,
      sumSq = 0,
      count = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const v =
          gray[idx - w] +
          gray[idx - 1] -
          4 * gray[idx] +
          gray[idx + 1] +
          gray[idx + w];
        sum += v;
        sumSq += v * v;
        count++;
      }
    }

    const mean = sum / (count || 1);
    const variance = sumSq / (count || 1) - mean * mean;
    return variance;
  }

  async function captureCandidateFrame() {
    const video = videoRef.current;
    const box = selectedBoxRef.current;
    if (!video || !box) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    let x1 = Math.floor(box.minX * vw);
    let y1 = Math.floor(box.minY * vh);
    let x2 = Math.ceil(box.maxX * vw);
    let y2 = Math.ceil(box.maxY * vh);

    const padX = Math.floor((x2 - x1) * 0.55);
    const padY = Math.floor((y2 - y1) * 0.65);
    x1 = Math.max(0, x1 - padX);
    y1 = Math.max(0, y1 - padY);
    x2 = Math.min(vw, x2 + padX);
    y2 = Math.min(vh, y2 + padY);

    const cw = Math.max(1, x2 - x1);
    const ch = Math.max(1, y2 - y1);

    const cap = document.createElement("canvas");
    cap.width = cw;
    cap.height = ch;

    const ctx = cap.getContext("2d")!;
    ctx.drawImage(video, x1, y1, cw, ch, 0, 0, cw, ch);

    const blurScore = computeBlurScoreFromCanvas(cap);
    const blob = await new Promise<Blob | null>((resolve) =>
      cap.toBlob(resolve, "image/jpeg", 0.95),
    );
    if (!blob) return;

    candidatesRef.current.push({ blob, blurScore });
    if (candidatesRef.current.length > CANDIDATE_BUFFER_MAX)
      candidatesRef.current.shift();
  }

  function pickTopCandidates(k = TEMPLATES_TO_SEND) {
    const arr = candidatesRef.current.slice();
    if (!arr.length) return [];
    arr.sort((a, b) => b.blurScore - a.blurScore);
    return arr.slice(0, k);
  }

  // ✅ same logic, ONLY text changed (no "MRZ" mention)
  function toUiResult(resStatus: number, detail: any): UiResult {
    const r = detail?.reason;

    if (resStatus === 409) {
      if (r === "person_id_exists") {
        return {
          kind: "warn",
          title: "رقم الوثيقة مستخدم مسبقاً",
          desc: "يرجى الرجوع وتعديل رقم الوثيقة ثم إعادة المحاولة.",
          action: "back_mrz",
        };
      }

      if (r === "face_already_assigned") {
        return {
          kind: "warn",
          title: "هذا الوجه مسجل مسبقاً",
          desc: "يرجى الرجوع وإعادة المحاولة من جديد.",
          action: "back_mrz",
        };
      }

      return {
        kind: "error",
        title: "تعذر إتمام العملية",
        desc: "حدث تعارض. يرجى الرجوع وإعادة المحاولة.",
        action: "back_mrz",
      };
    }

    if (resStatus === 403) {
      return {
        kind: "error",
        title: "فشل التحقق الحيوي",
        desc: "يرجى الرجوع وإعادة المحاولة.",
        action: "back_mrz",
      };
    }

    if (resStatus === 400) {
      return {
        kind: "error",
        title: "تعذر التقاط صورة مناسبة",
        desc: "يرجى الرجوع وإعادة المحاولة بصورة أوضح.",
        action: "back_mrz",
      };
    }

    return {
      kind: "error",
      title: "خطأ في الخادم",
      desc: "يرجى الرجوع وإعادة المحاولة.",
      action: "back_mrz",
    };
  }

  async function sendEnrollWithBlobs(blobs: Blob[]) {
    if (inFlightRef.current) return;

    const now = Date.now();
    if (now - lastSentAtRef.current < COOLDOWN_MS) return;

    if (!personId) {
      setUiResult({
        kind: "warn",
        title: "رقم الوثيقة غير موجود",
        desc: "يرجى الرجوع وتعديل رقم الوثيقة.",
        action: "back_mrz",
      });
      setStep("result");
      return;
    }

    inFlightRef.current = true;
    lastSentAtRef.current = now;

    setBusy(true);
    setUiResult(null);

    stopCamera();
    setStep("processing");

    try {
      const form = new FormData();
      form.append("person_id", personId);

      form.append("first_name", (info?.firstName || "").trim());
      form.append("last_name", (info?.lastName || "").trim());
      form.append(
        "email",
        (info as any)?.email ? String((info as any).email).trim() : "",
      );

      // images (unchanged)
      blobs.forEach((b, i) => form.append("images", b, `enroll_${i + 1}.jpg`));

      const res = await fetch(apiUrl("/face/enroll"), {
        method: "POST",
        body: form,
      });
      const data = await res.json().catch(() => ({}) as any);
      const detail = (data as any)?.detail;

      if (!res.ok) {
        setBusy(false);
        setUiResult(toUiResult(res.status, detail || data || {}));
        setStep("result");
        return;
      }

      setBusy(false);
      setUiResult({
        kind: "success",
        title: "تم تسجيل الوجه بنجاح",
        desc: "تم حفظ بيانات الوجه وربطها برقم الوثيقة.",
        action: "done",
      });
      setStep("result");
    } catch {
      setBusy(false);
      setUiResult({
        kind: "error",
        title: "تعذر الاتصال بالخادم",
        desc: "يرجى الرجوع وإعادة المحاولة.",
        action: "back_mrz",
      });
      setStep("result");
    } finally {
      setBusy(false);
      inFlightRef.current = false;
    }
  }

  async function loop() {
    if (step !== "camera") {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }

    const landmarker = landmarkerRef.current;
    const video = videoRef.current;

    if (!landmarker || !video || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }

    try {
      const detection = landmarker.detectForVideo(video, performance.now());
      const gate = gateSelectFaceInsideOval(detection);

      if (!gate.ok) {
        setStatus(gate.msg);
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const box = selectedBoxRef.current;
      if (!box) {
        setStatus("ضع وجهك داخل الإطار");
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const stable = isStable(box);
      if (!stable) {
        setStatus("ثبّت الهاتف");
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const nowMs = Date.now();
      if (nowMs - lastCaptureAtRef.current >= CAPTURE_EVERY_MS) {
        lastCaptureAtRef.current = nowMs;
        await captureCandidateFrame();
        framesRef.current += 1;
      }

      const p = Math.min(
        100,
        Math.round((framesRef.current / GOOD_FRAMES_REQUIRED) * 100),
      );
      setStatus(p < 100 ? "جاري الالتقاط…" : "جاري الإرسال…");

      if (framesRef.current >= GOOD_FRAMES_REQUIRED) {
        const top = pickTopCandidates(TEMPLATES_TO_SEND);
        const blobs = top.map((x) => x.blob).filter(Boolean);
        if (!inFlightRef.current && blobs.length) {
          await sendEnrollWithBlobs(blobs);
        }
      }
    } catch {
      setStatus("تعذر الكشف");
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  async function startCamera() {
    resetRuntime();
    stopCamera();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    streamRef.current = stream;
    const video = videoRef.current!;
    video.srcObject = stream;

    await new Promise<void>((res) => {
      video.onloadedmetadata = () => res();
    });

    await video.play();

    const canvas = overlayRef.current!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    setStatus("ضع وجهك داخل الإطار");
    stopLoop();
    rafRef.current = requestAnimationFrame(loop);
  }

  async function init() {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
    );

    landmarkerRef.current = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
    });

    stopLoop();
    rafRef.current = requestAnimationFrame(loop);
  }

  // ✅ ONLY button labels changed (no "MRZ" mention)
  function primaryActionLabel() {
    if (!uiResult) return "رجوع لإعادة المحاولة";
    if (uiResult.action === "back_mrz") return "رجوع لإعادة المحاولة";
    return "إنهاء";
  }

  function handlePrimaryAction() {
    if (!uiResult) return;

    if (uiResult.action === "back_mrz") {
      stopCamera();
      onBackToMrz();
      return;
    }

    stopCamera();
    onDone();
  }

  useEffect(() => {
    init().catch(() => {});
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    startCamera().catch(() => setStatus("تعذر تشغيل الكاميرا"));
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section className="screen">
      <div className="screen-body">
        <h1 className="screen-title">تسجيل الوجه</h1>
        <p className="screen-subtitle">
          ضع وجهك داخل الإطار وسيتم الالتقاط تلقائياً
        </p>

        {step === "camera" && (
          <div className="ovalStage">
            <div
              className={`ovalWindow ${status.includes("جاري") ? "good" : ""}`}
            >
              <div
                className={`topPill ${status.includes("جاري") ? "good" : ""}`}
              >
                {status}
              </div>

              <video ref={videoRef} className="ovalVideo" playsInline />
              <canvas ref={overlayRef} className="ovalCanvas" />
            </div>
          </div>
        )}

        {step === "processing" && (
          <div className="result-card">
            <div className="result-badge warn">…</div>
            <div className="result-title">جاري التحقق</div>
            <div className="result-desc">الرجاء الانتظار</div>
          </div>
        )}

        {step === "result" && uiResult && (
          <div className="result-card">
            <div className={`result-badge ${uiResult.kind}`}>
              {uiResult.kind === "success"
                ? "✓"
                : uiResult.kind === "warn"
                  ? "!"
                  : "×"}
            </div>
            <div className="result-title">{uiResult.title}</div>
            <div className="result-desc">{uiResult.desc}</div>
          </div>
        )}
      </div>

      <div className="screen-actions">
        {step === "camera" && (
          <div className="actions-row">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                stopCamera();
                onBack();
              }}
              disabled={busy}
            >
              رجوع
            </button>

            <button
              className="primary-button"
              type="button"
              onClick={() => {
                stopCamera();
                onBackToMrz();
              }}
              disabled={busy}
            >
              رجوع لإعادة المحاولة
            </button>
          </div>
        )}

        {step === "processing" && (
          <div className="actions-row">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                stopCamera();
                onBack();
              }}
              disabled={busy}
            >
              رجوع
            </button>

            <button
              className="primary-button"
              type="button"
              onClick={() => {
                stopCamera();
                onBackToMrz();
              }}
              disabled={busy}
            >
              رجوع لإعادة المحاولة
            </button>
          </div>
        )}

        {step === "result" && uiResult && (
          <div className="actions-row single">
            <button
              className="primary-button"
              type="button"
              onClick={handlePrimaryAction}
              disabled={busy}
            >
              {primaryActionLabel()}
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
