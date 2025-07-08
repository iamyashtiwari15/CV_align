/*
import { useState, ChangeEvent } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { evaluateCV } from "@/utils/api"; // ✅ Import the API call
import { Typewriter } from "react-simple-typewriter"; // remove this later

const CVUpload = () => {
  const [jobDescription, setJobDescription] = useState("");
  const [cvFile, setCvFile] = useState<File | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<string>(""); //remove this later
  const [showAnalysis, setShowAnalysis] = useState<boolean>(false);// remove this later
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCvFile(e.target.files[0]);
    }
  };

const handleSubmit = async () => {
  if (!cvFile || !jobDescription) return;
  setLoading(true);

  try {
    const result = await evaluateCV(cvFile, jobDescription);
    setFeedback(result.answer); // because your backend returns { answer: ... }
  } catch (error) {
    console.error("Error:", error);
  } finally {
    setLoading(false);
  }
};

  return (
    <Card className="mt-4">
      <CardContent className="space-y-4">
        <Textarea
          placeholder="Enter Job Description"
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />
        <Input type="file" accept=".pdf,.docx" onChange={handleUpload} />
        {cvFile && <p className="text-sm text-gray-500">File: {cvFile.name}</p>}
        <Button onClick={handleSubmit} disabled={loading}>
          {loading ? "Evaluating..." : "Submit"}
        </Button>

        {feedback !== null && (
          <div className="space-y-2 pt-4">
            <p className="text-lg font-semibold">{score}</p>
            <p className="whitespace-pre-wrap">{feedback}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CVUpload;
*/
import { useState, ChangeEvent } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { evaluateCV } from "@/utils/api";
import { Typewriter } from "react-simple-typewriter";

const CVUpload = () => {
  const [jobDescription, setJobDescription] = useState("");
  const [cvFile, setCvFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<string>("");
  const [showAnalysis, setShowAnalysis] = useState<boolean>(false);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCvFile(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!cvFile || !jobDescription) return;
    setLoading(true);
    setShowAnalysis(false);

    try {
      const result = await evaluateCV(cvFile, jobDescription);
      setAnalysis(result.answer); // Store full result
      setShowAnalysis(true); // Trigger typewriter animation
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="mt-4">
      <CardContent className="space-y-4">
        <Textarea
          placeholder="Enter Job Description"
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />
        <Input type="file" accept=".pdf,.docx" onChange={handleUpload} />
        {cvFile && <p className="text-sm text-gray-500">File: {cvFile.name}</p>}
        <Button onClick={handleSubmit} disabled={loading}>
          {loading ? "Evaluating..." : "Submit"}
        </Button>

        {showAnalysis && (
          <div className="mt-6 p-4 border rounded-lg bg-gray-100 text-sm font-mono text-gray-800 whitespace-pre-wrap">
            <Typewriter
              words={[analysis]}
              cursor
              typeSpeed={25}
              deleteSpeed={40}
              delaySpeed={2000}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CVUpload;
