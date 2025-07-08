/*
// src/pages/RecruiterForm.tsx
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const RecruiterForm = () => {
  const [jobTitle, setJobTitle] = useState("");
  const [requiredSkills, setRequiredSkills] = useState("");
  const [preferredQualifications, setPreferredQualifications] = useState("");
  const [jobDescription, setJobDescription] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const payload = {
      jobTitle,
      requiredSkills,
      preferredQualifications,
      jobDescription,
    };

    console.log("Submitting Job Role:", payload);
    // Here you can POST to your FastAPI backend
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 p-4">
      <Card className="w-full max-w-xl shadow-lg p-6">
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <h2 className="text-2xl font-semibold mb-4">Create Job Role</h2>

            <div>
              <label className="block text-sm font-medium mb-1">Job Title</label>
              <Input
                placeholder="e.g., Machine Learning Engineer"
                value={jobTitle}
                onChange={(e) => setJobTitle(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Required Skills</label>
              <Input
                placeholder="e.g., Python, TensorFlow, NLP"
                value={requiredSkills}
                onChange={(e) => setRequiredSkills(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Preferred Qualifications</label>
              <Input
                placeholder="e.g., 2+ years experience, GitHub portfolio"
                value={preferredQualifications}
                onChange={(e) => setPreferredQualifications(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Job Description</label>
              <Textarea
                placeholder="Describe the role and responsibilities..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
              />
            </div>

            <Button type="submit" className="w-full">
              Submit Job Role
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default RecruiterForm;
*/

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Typewriter } from "react-simple-typewriter";

const RecruiterForm = () => {
  const [jobTitle, setJobTitle] = useState("");
  const [requiredSkills, setRequiredSkills] = useState("");
  const [preferredQualifications, setPreferredQualifications] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [showCVResults, setShowCVResults] = useState(false);
  const [cvResults, setCvResults] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Simulated AI output
    const simulatedOutput = `1. CV(218) – 85% Match (Strong Fit)
Name: Rajesh Kumar
Contact: +91 98765 43210 | rajesh.kumar@vectordb.ai
Background: ML Engineer, 4 years at AI startup
Key Skills:
FAISS (IVF, HNSW, GPU-accelerated), ChromaDB (metadata filtering)
Deployed RAG pipelines for enterprise chatbots
AWS SageMaker + custom embedding models (Sentence-BERT fine-tuning)
Gap: No billion-scale vector experience

2. CV(572) – 78% Match (Production Focus)
Name: Elena Rodriguez
Contact: +34 600 112 233 | elena.rodriguez@mlops.es
Background: MLOps Engineer, ex-FAANG
Key Skills:
ChromaDB cloud deployment (Docker, Kubernetes)
FAISS for real-time recommender systems (10M+ vectors)
Hybrid search (keyword + semantic) with ElasticSearch integration
Gap: Theoretical ANN knowledge (e.g., PQ optimizations)

3. CV(754) – 72% Match (Research-Leaning)
Name: Dr. James Wu
Contact: +1 (650) 555-0199 | james.wu@stanford.edu
Background: PhD in NLP, Postdoc at MIT
Key Skills:
FAISS expert (published papers on IVF-HNSW tradeoffs)
Custom embedding models (contrastive learning)
LangChain + ChromaDB prototypes
Gap: Limited production scaling experience

4. CV(63) – 65% Match (Career Transitioner)
Name: Aisha Mohammed
Contact: +44 7788 123456 | aisha.m@careerswitch.uk
Background: Data Scientist (2 years) + Coursera ML certifications
Key Skills:
ChromaDB for document retrieval (side project)
FAISS benchmarks on 1M vectors (Python)
Basic RAG with OpenAI embeddings
Gap: No cloud deployment or metadata filtering

5. CV(901) – 60% Match (Cloud Specialist)
Name: Derek Holt
Contact: +1 (415) 555-6789 | derek.holt@cloudguru.com
Background: Cloud Architect, AWS/Azure certified
Key Skills:
Deployed ChromaDB on AWS EKS (proof-of-concept)
FAISS via PyTorch (beginner-level)
Strong infrastructure scaling knowledge
Gap: No embedding model fine-tuning`;

    setCvResults(simulatedOutput);
    setShowCVResults(true);
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 p-4">
      <Card className="w-full max-w-xl shadow-lg p-6">
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <h2 className="text-2xl font-semibold mb-4">Create Job Role</h2>

            <div>
              <label className="block text-sm font-medium mb-1">Job Title</label>
              <Input
                placeholder="e.g., Machine Learning Engineer"
                value={jobTitle}
                onChange={(e) => setJobTitle(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Required Skills</label>
              <Input
                placeholder="e.g., Python, TensorFlow, NLP"
                value={requiredSkills}
                onChange={(e) => setRequiredSkills(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Preferred Qualifications</label>
              <Input
                placeholder="e.g., 2+ years experience, GitHub portfolio"
                value={preferredQualifications}
                onChange={(e) => setPreferredQualifications(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Job Description</label>
              <Textarea
                placeholder="Describe the role and responsibilities..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
              />
            </div>

            <Button type="submit" className="w-full">
              Submit Job Role
            </Button>
          </form>

          {showCVResults && (
            <div className="mt-6 p-4 bg-gray-100 rounded-lg text-sm font-mono whitespace-pre-wrap text-gray-800">
              <Typewriter
                words={[cvResults]}
                cursor
                typeSpeed={25}
                delaySpeed={1000000}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default RecruiterForm;

