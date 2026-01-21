import { useEffect, useMemo, useState } from "react";
import { extractPersonalInfoFromImage } from "../mrzUtils";
import type { PersonalInfo } from "../types";

interface CameraScannerProps {
  onSuccess: (data: { info: PersonalInfo; faceDataUrl?: string }) => void;
  onError: (message: string) => void;
  onStartExtract: () => void;
}

export function CameraScanner({
  onSuccess,
  onError,
  onStartExtract,
}: CameraScannerProps) {
  const [isProcessing, setIsProcessing] = useState(false);

  // ✅ rotating button text while processing (keeps user engaged)
  const processingTexts = useMemo(
    () => [
      "جارٍ قراءة الصورة…",
      "جارٍ تحسين الجودة…",
      "جارٍ استخراج البيانات…",
      "جارٍ التحقق من النتائج…",
    ],
    [],
  );
  const [procTextIndex, setProcTextIndex] = useState(0);

  useEffect(() => {
    if (!isProcessing) return;
    setProcTextIndex(0);
    const id = window.setInterval(() => {
      setProcTextIndex((i) => (i + 1) % processingTexts.length);
    }, 2000);
    return () => window.clearInterval(id);
  }, [isProcessing, processingTexts.length]);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      onError("الرجاء اختيار ملف صورة صالح.");
      return;
    }

    setIsProcessing(true);
    onStartExtract();

    try {
      const reader = new FileReader();

      const dataUrl: string = await new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });

      const result = await extractPersonalInfoFromImage(dataUrl);

      if (!result?.info) {
        onError("تعذر قراءة المعلومات من الصورة. حاول استخدام صورة أوضح.");
        return;
      }

      onSuccess({
        info: result.info,
        faceDataUrl: result.faceDataUrl,
      });
    } catch (error) {
      console.error("Error in handleFileChange:", error);
      onError("حدث خطأ أثناء معالجة الصورة.");
    } finally {
      setIsProcessing(false);
      event.target.value = "";
    }
  };

  return (
    <div className="mrz-block">
      <div className="mrz-frame mrz-corners">
        <div className="mrz-card-illustration">
          {isProcessing && (
            <div className="mrz-processing-overlay">
              <div className="mrz-dim" />
              <div className="mrz-scan-line" />
            </div>
          )}
        </div>
      </div>

      <ul className="mrz-tips">
        <li>ارفع صورة واضحة ومواجهة للبطاقة</li>
        <li>اجعل البطاقة مستقيمة داخل الصورة</li>
        <li>تجنب الصور المائلة أو بزاوية</li>
        <li>استخدم إضاءة جيدة بدون انعكاسات</li>
      </ul>

      <div className="mrz-upload-wrap">
        <label className="primary-button upload-button">
          {isProcessing
            ? processingTexts[procTextIndex]
            : "اختيار صورة من الجهاز"}
          <input
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFileChange}
            disabled={isProcessing}
          />
        </label>
      </div>
    </div>
  );
}
