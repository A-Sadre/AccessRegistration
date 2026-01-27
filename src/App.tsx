import "./App.css";
import { useEffect, useMemo, useState } from "react";
import { CameraScanner } from "./components/CameraScanner";
import { FaceEnroll } from "./components/FaceEnroll";
import type { PersonalInfo } from "./types";

type Step = "intro" | 1 | 2 | 3 | 4 | 5;
const TOTAL_STEPS = 5;

type FormState = PersonalInfo & { email: string };

function IconShield() {
  return (
    <svg viewBox="0 0 24 24" className="cardIconSvg" aria-hidden="true">
      <path
        d="M12 2l8 4v6c0 5-3.4 9.4-8 10-4.6-.6-8-5-8-10V6l8-4z"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="M9.5 12l1.8 1.8 3.8-3.9"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg viewBox="0 0 24 24" className="cardIconSvg" aria-hidden="true">
      <path
        d="M20 6L9 17l-5-5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconBolt() {
  return (
    <svg viewBox="0 0 24 24" className="cardIconSvg" aria-hidden="true">
      <path
        d="M13 2L3 14h7l-1 8 12-14h-7l-1-6z"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function StepsHeader({ step }: { step: Step }) {
  if (step === "intro") return null;

  const current = step;

  return (
    <header className="phone-header">
      <div className="steps-row">
        <div className="steps-label">الخطوات</div>
        <div className="steps-count" dir="ltr">
          {current} / {TOTAL_STEPS}
        </div>
      </div>

      <div className="steps-segments" aria-label="progress">
        {Array.from({ length: TOTAL_STEPS }).map((_, i) => {
          const segIndex = i + 1;
          const isActive = current >= segIndex;
          return <span key={segIndex} className={`seg ${isActive ? "active" : ""}`} />;
        })}
      </div>
    </header>
  );
}

function isEmpty(v: string | undefined | null) {
  return !String(v ?? "").trim();
}

function isValidEmail(email: string) {
  const v = (email || "").trim();
  if (!v) return false;
  return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v);
}

export default function App() {
  const [step, setStep] = useState<Step>("intro");
  const [info, setInfo] = useState<PersonalInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [faceImage, setFaceImage] = useState<string | null>(null);

  const [form, setForm] = useState<FormState | null>(null);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  // ✅ rotating messages for Step 3 (to keep user engaged)
  const processingMessages = useMemo(
    () => [
      " تحسين وضوح النص…",
      " تحديد منطقة القراءة…",
      " استخراج البيانات…",
      " التحقق من صحة الحروف…",
      " مطابقة الحقول…",
      " تجهيز المعلومات للعرض…",
    ],
    [],
  );

  const [processingMsgIndex, setProcessingMsgIndex] = useState(0);

  useEffect(() => {
    if (step !== 3) return;

    setProcessingMsgIndex(0);
    const id = window.setInterval(() => {
      setProcessingMsgIndex((i) => (i + 1) % processingMessages.length);
    }, 1500);

    return () => window.clearInterval(id);
  }, [step, processingMessages.length]);

  const handleMrzDetected = (data: { info: PersonalInfo; faceDataUrl?: string }) => {
    setInfo(data.info);
    setFaceImage(data.faceDataUrl || null);
    setError(null);

    setForm({
      ...data.info,
      email: "",
    });
    setFormErrors({});
    setStep(4);
  };

  const handleStartExtract = () => {
    setError(null);
    setStep(3);
  };

  function validateStep4(f: FormState) {
    const errs: Record<string, string> = {};

    if (isEmpty(f.firstName)) errs.firstName = "هذا الحقل مطلوب";
    if (isEmpty(f.lastName)) errs.lastName = "هذا الحقل مطلوب";
    if (isEmpty(f.documentNumber)) errs.documentNumber = "هذا الحقل مطلوب";
    if (isEmpty(f.email)) errs.email = "هذا الحقل مطلوب";
    else if (!isValidEmail(f.email)) errs.email = "البريد الإلكتروني غير صالح";

    if (isEmpty(f.nationality)) errs.nationality = "هذا الحقل مطلوب";
    if (isEmpty(f.dateOfBirth)) errs.dateOfBirth = "هذا الحقل مطلوب";
    if (isEmpty(f.sex)) errs.sex = "هذا الحقل مطلوب";
    if (isEmpty(f.expirationDate)) errs.expirationDate = "هذا الحقل مطلوب";

    return errs;
  }

  function confirmStep4() {
    if (!form) return;

    const errs = validateStep4(form);
    setFormErrors(errs);

    if (Object.keys(errs).length) {
      setError("يرجى تعبئة جميع الحقول المطلوبة قبل المتابعة.");
      return;
    }

    const { email: _email, ...personal } = form;
    setInfo(personal);
    setError(null);
    setStep(5);
  }

  return (
    <div className="app-root">
      <div className="phone-shell" dir="rtl" lang="ar">
        <StepsHeader step={step} />

        <main className="phone-content">
          {/* INTRO */}
          {step === "intro" && (
            <section className="screen">
              <div className="screen-body screen-body--design screen-body--intro">
                <div className="screen-top screen-top--spaced">
                  <h1 className="screen-title">التحقق الرقمي من الهوية</h1>

                  <p className="screen-subtitle screen-subtitle--hero screen-subtitle--gap">
                    تحقق من هويتك بأمان باستخدام جواز السفر لدخول آمن لمنشآتنا
                  </p>
                </div>

                <div className="feature-cards feature-cards--hero feature-cards--gap">
                  <div className="feature-card feature-blue feature-card--hero">
                    <div className="feature-card-icon blue">
                      <IconShield />
                    </div>
                    <div className="feature-card-text">
                      <div className="feature-card-title">بيانات مشفرة</div>
                      <div className="feature-card-sub">جميع البيانات محمية ومشفرة</div>
                    </div>
                  </div>

                  <div className="feature-card feature-amber feature-card--hero">
                    <div className="feature-card-icon amber">
                      <IconCheck />
                    </div>
                    <div className="feature-card-text">
                      <div className="feature-card-title">متوافق مع الأنظمة</div>
                      <div className="feature-card-sub">مناسب للاستخدام في بيئات رسمية</div>
                    </div>
                  </div>

                  <div className="feature-card feature-red feature-card--hero">
                    <div className="feature-card-icon red">
                      <IconBolt />
                    </div>
                    <div className="feature-card-text">
                      <div className="feature-card-title">إجراء سريع</div>
                      <div className="feature-card-sub">يستغرق أقل من 3 دقائق</div>
                    </div>
                  </div>
                </div>

                <div className="screen-spacer" />
              </div>

              <div className="screen-actions screen-actions--raised">
                <button
                  className="primary-button primary-button--big"
                  type="button"
                  onClick={() => setStep(1)}
                >
                  بدء التسجيل
                </button>
              </div>
            </section>
          )}

          {/* STEP 1 */}
          {step === 1 && (
            <section className="screen">
              <div className="screen-body screen-body--design">
                <div className="screen-top screen-top--spaced">
                  <h1 className="screen-title">قبل البدء</h1>
                  <p className="screen-subtitle screen-subtitle--gap">تأكد من تحقق هذه الشروط</p>
                </div>

                <div className="check-cards">
                  <div className="check-item">
                    <span className="check-item-icon">
                      <IconCheck />
                    </span>
                    <span className="check-item-text">بطاقة جواز السفر الأصلي</span>
                  </div>

                  <div className="check-item">
                    <span className="check-item-icon">
                      <IconCheck />
                    </span>
                    <span className="check-item-text">مكان مضاء جيداً</span>
                  </div>

                  <div className="check-item">
                    <span className="check-item-icon">
                      <IconCheck />
                    </span>
                    <span className="check-item-text">كاميرا الهاتف نظيفة وواضحة</span>
                  </div>
                </div>

                <div className="screen-spacer" />
              </div>

              <div className="screen-actions screen-actions--raised">
                <button
                  className="primary-button primary-button--big"
                  type="button"
                  onClick={() => setStep(2)}
                >
                  متابعة
                </button>
              </div>
            </section>
          )}

          {/* STEP 2 */}
          {step === 2 && (
            <section className="screen">
              <div className="screen-body screen-body--design">
                <div className="screen-top screen-top--spaced">
                  <h1 className="screen-title">تصوير جواز السفر </h1>
                  <p className="screen-subtitle screen-subtitle--gap">
                    قم برفع صورة واضحة بحيث تظهر منطقة MRZ
                  </p>
                </div>

                <div className="mrz-center mrz-center--hero">
                  <CameraScanner
                    onSuccess={handleMrzDetected}
                    onError={(msg) => {
                      setError(msg);
                      setStep(2);
                    }}
                    onStartExtract={handleStartExtract}
                  />
                </div>

                {error && <div className="alert alert-danger">{error}</div>}

                <div className="screen-spacer" />
              </div>

              <div className="screen-actions screen-actions--ghost" />
            </section>
          )}

          {/* STEP 3 (PROCESSING) */}
          {step === 3 && (
            <section className="screen">
              <div className="screen-body screen-body--design">
                <div className="screen-top screen-top--spaced">
                  <h1 className="screen-title">استخراج البيانات</h1>
                  <p className="screen-subtitle screen-subtitle--gap">
                    الرجاء الانتظار… يتم الآن قراءة البيانات
                  </p>
                </div>

                <div className="extract-card extract-card--hero">
                  <div className="extract-illustration extract-illustration--scan">
                    <div className="mrz-card-illustration mrz-card-illustration--photo">
                      <div className="mrz-processing-overlay">
                        <div className="mrz-dim" />
                        <div className="mrz-scan-line" />
                      </div>
                    </div>
                  </div>

                  <div className="loading-row">
                    <span className="loading-text">
                      {processingMessages[processingMsgIndex]}
                    </span>
                    <span className="loading-spinner" />
                  </div>
                </div>

                <div className="screen-spacer" />
              </div>

              <div className="screen-actions no-border" />
            </section>
          )}

          {/* STEP 4 */}
          {step === 4 && form && (
            <section className="screen">
              <div className="screen-body">
                <div className="banner banner-success">
                  <div className="banner-icon">✓</div>
                  <div className="banner-text">تم استخراج البيانات تلقائياً من الهوية</div>
                </div>

                <h1 className="screen-title">التحقق من صحة المعلومات</h1>
                <p className="screen-subtitle">قم بمراجعة المعلومات وتعديلها عند الحاجة</p>

                {/* {faceImage && (
                  <div className="face-preview-wrap">
                    <img className="face-preview" src={faceImage} alt="الوجه" />
                  </div>
                )} */}

                {error && <div className="alert alert-danger">{error}</div>}

                <form
                  className="form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    confirmStep4();
                  }}
                >
                  <div className="field">
                    <label className="field-label">الاسم الشخصي *</label>
                    <input
                      className={`field-input ${formErrors.firstName ? "field-input--error" : ""}`}
                      value={form.firstName}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, firstName: "" }));
                        setForm({ ...form, firstName: e.target.value });
                      }}
                      required
                    />
                    {formErrors.firstName && <div className="field-error">{formErrors.firstName}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">الاسم العائلي *</label>
                    <input
                      className={`field-input ${formErrors.lastName ? "field-input--error" : ""}`}
                      value={form.lastName}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, lastName: "" }));
                        setForm({ ...form, lastName: e.target.value });
                      }}
                      required
                    />
                    {formErrors.lastName && <div className="field-error">{formErrors.lastName}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">رقم الوثيقة *</label>
                    <input
                      className={`field-input ${formErrors.documentNumber ? "field-input--error" : ""}`}
                      value={form.documentNumber}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, documentNumber: "" }));
                        setForm({ ...form, documentNumber: e.target.value });
                      }}
                      required
                    />
                    {formErrors.documentNumber && <div className="field-error">{formErrors.documentNumber}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">البريد الإلكتروني *</label>
                    <input
                      className={`field-input ${formErrors.email ? "field-input--error" : ""}`}
                      value={form.email}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, email: "" }));
                        setForm({ ...form, email: e.target.value });
                      }}
                      type="email"
                      inputMode="email"
                      autoComplete="email"
                      placeholder="example@mail.com"
                      required
                    />
                    {formErrors.email && <div className="field-error">{formErrors.email}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">الجنسية *</label>
                    <input
                      className={`field-input ${formErrors.nationality ? "field-input--error" : ""}`}
                      value={form.nationality}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, nationality: "" }));
                        setForm({ ...form, nationality: e.target.value });
                      }}
                      required
                    />
                    {formErrors.nationality && <div className="field-error">{formErrors.nationality}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">تاريخ الميلاد *</label>
                    <input
                      className={`field-input ${formErrors.dateOfBirth ? "field-input--error" : ""}`}
                      type="date"
                      value={form.dateOfBirth}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, dateOfBirth: "" }));
                        setForm({ ...form, dateOfBirth: e.target.value });
                      }}
                      required
                    />
                    {formErrors.dateOfBirth && <div className="field-error">{formErrors.dateOfBirth}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">الجنس *</label>
                    <input
                      className={`field-input ${formErrors.sex ? "field-input--error" : ""}`}
                      value={form.sex}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, sex: "" }));
                        setForm({ ...form, sex: e.target.value });
                      }}
                      required
                    />
                    {formErrors.sex && <div className="field-error">{formErrors.sex}</div>}
                  </div>

                  <div className="field">
                    <label className="field-label">انتهاء الوثيقة *</label>
                    <input
                      className={`field-input ${formErrors.expirationDate ? "field-input--error" : ""}`}
                      type="date"
                      value={form.expirationDate}
                      onChange={(e) => {
                        setFormErrors((p) => ({ ...p, expirationDate: "" }));
                        setForm({ ...form, expirationDate: e.target.value });
                      }}
                      required
                    />
                    {formErrors.expirationDate && <div className="field-error">{formErrors.expirationDate}</div>}
                  </div>
                </form>
              </div>

              <div className="screen-actions">
                <div className="actions-row">
                  <button className="secondary-button" type="button" onClick={() => setStep(2)}>
                    إلغاء
                  </button>
                  <button className="primary-button" type="button" onClick={confirmStep4}>
                    تأكيد
                  </button>
                </div>
              </div>
            </section>
          )}

          {/* STEP 5 */}
          {step === 5 && info && (
            <FaceEnroll
              info={info}
              onDone={() => {
                alert("تم التسجيل بالكامل ✅");
                setStep("intro");
                setInfo(null);
                setForm(null);
                setFormErrors({});
                setFaceImage(null);
                setError(null);
              }}
              onBack={() => setStep(4)}
              onBackToMrz={() => setStep(2)}
            />
          )}
        </main>

        <footer className="phone-footer">
          <span className="footer-check">🔒</span>
          يتم تشفير جميع البيانات وفق أعلى معايير الأمان ولا يتم تخزينها دون موافقتك
        </footer>
      </div>
    </div>
  );
}
