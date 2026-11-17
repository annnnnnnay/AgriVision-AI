import { motion } from "framer-motion";
import {
  Github,
  ExternalLink,
  Cpu,
  Server,
  Cloud,
  Layers,
} from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 10 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.05, duration: 0.3, ease: "easeOut" },
  }),
};

const techStack = [
  {
    category: "AI & Machine Learning",
    icon: Cpu,
    items: [
      { name: "Gemini 3.8 Flash", desc: "Primary cloud vision provider" },
      { name: "OpenRouter Vision", desc: "Secondary cloud vision provider" },
      { name: "MobileNetV2 (Keras)", desc: "Offline, leaf-only 46-class classifier" },
    ],
  },
  {
    category: "Backend & Database",
    icon: Server,
    items: [
      { name: "FastAPI", desc: "Async Python web framework" },
      { name: "Supabase (PostgreSQL)", desc: "Relational database with RLS policies" },
      { name: "Supabase Storage", desc: "Blob storage for uploaded crop photos" },
    ],
  },
  {
    category: "Frontend",
    icon: Layers,
    items: [
      { name: "React 18", desc: "UI library" },
      { name: "Tailwind CSS", desc: "Premium responsive UI design" },
      { name: "Framer Motion", desc: "Micro-animations & transitions" },
    ],
  },
  {
    category: "Infrastructure",
    icon: Cloud,
    items: [
      { name: "JWT Security", desc: "Secure token-based user authentication" },
      { name: "Dynamic Local KB", desc: "Rule-based multi-language fallback" },
      { name: "Docker", desc: "Containerization" },
    ],
  },
];

export default function About() {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div initial="hidden" animate="visible" variants={fadeUp}>
          <h1 className="font-display text-3xl sm:text-4xl font-bold text-slate-900">
            About <span className="gradient-text">AgriVision AI</span>
          </h1>
          <p className="mt-3 text-lg text-slate-600 leading-relaxed max-w-2xl">
            AgriVision AI is a hybrid AI-powered crop disease detection platform
            built to help farmers identify plant diseases instantly
            using a smartphone camera — online or offline.
          </p>
        </motion.div>

        {/* Mission */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="glass-card p-8 mt-10"
        >
          <h2 className="font-display text-xl font-bold text-slate-900 mb-4">
            🎯 Our Mission
          </h2>
          <p className="text-slate-600 leading-relaxed">
            Indian agriculture loses over ₹90,000 crores annually to crop
            diseases. Farmers in rural areas lack access to timely diagnosis
            and expert advice. AgriVision AI bridges this gap by putting the
            power of artificial intelligence in every farmer's pocket. Our 
            three-tier hybrid architecture tries configured cloud vision providers
            before falling back to an offline leaf classifier.
          </p>
          <div className="grid sm:grid-cols-3 gap-4 mt-6">
            <div className="text-center p-4 bg-brand-50 rounded-xl">
              <p className="text-2xl font-bold text-brand-700">46</p>
              <p className="text-sm text-brand-600">Local Leaf Classes</p>
            </div>
            <div className="text-center p-4 bg-brand-50 rounded-xl">
              <p className="text-2xl font-bold text-brand-700">3</p>
              <p className="text-sm text-brand-600">Inference Tiers</p>
            </div>
            <div className="text-center p-4 bg-brand-50 rounded-xl">
              <p className="text-2xl font-bold text-brand-700">3</p>
              <p className="text-sm text-brand-600">Languages (EN/HI/MR)</p>
            </div>
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="mt-10"
        >
          <h2 className="font-display text-2xl font-bold text-slate-900 mb-6">
            🛠️ Technology Stack
          </h2>
          <div className="grid sm:grid-cols-2 gap-4">
            {techStack.map(({ category, icon: Icon, items }, i) => (
              <motion.div
                key={category}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                custom={i}
                variants={fadeUp}
                className="glass-card p-6"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-brand-50 text-brand-600 flex items-center justify-center">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="font-semibold text-slate-900">{category}</h3>
                </div>
                <ul className="space-y-2">
                  {items.map(({ name, desc }) => (
                    <li
                      key={name}
                      className="flex items-center gap-2 text-sm"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                      <span className="font-medium text-slate-700">
                        {name}
                      </span>
                      <span className="text-slate-400">—</span>
                      <span className="text-slate-500">{desc}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* System Architecture */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="glass-card p-8 mt-10"
        >
          <h2 className="font-display text-xl font-bold text-slate-900 mb-4">
            🧠 System Architecture (3-Tier Cascade)
          </h2>
          <div className="font-mono text-xs bg-slate-900 text-emerald-400 p-6 rounded-xl overflow-x-auto leading-relaxed">
            <pre>{`Input Image + optional part/crop hint
                  ↓
       [Tier 1: Direct Gemini 3.8 Flash API] ──► (Accepted result) ──► Diagnosis (JSON)
                  ↓ (Fails / Rate Limit)
       [Tier 2: OpenRouter API Gateway]      ──► (Accepted result) ──► Diagnosis (JSON)
                  ↓ (Fails / Offline)
       [Tier 3: Local MobileNetV2 Model]     ──► Leaf selected only ──► 46 leaf classes
                  ↓ (Low confidence / non-leaf)
                  Decline instead of guessing`}</pre>
          </div>
          <p className="text-sm text-slate-500 mt-4">
            Cloud vision providers are attempted first. The bundled local model is trained on leaf-disease classes only and is used only when Leaf is selected; it declines low-confidence predictions and never substitutes a leaf label for a fruit or unspecified image.
          </p>
        </motion.div>

        {/* GitHub CTA */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="text-center mt-12"
        >
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary inline-flex items-center gap-2 text-base px-8 py-4"
          >
            <Github className="w-5 h-5" />
            View Source on GitHub
            <ExternalLink className="w-4 h-4" />
          </a>
        </motion.div>
      </div>
    </div>
  );
}
