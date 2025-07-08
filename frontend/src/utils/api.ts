/*export interface EvaluationResponse {
  answer: string;
}

export const evaluateCV = async (
  cvFile: File,
  jobDescription: string
): Promise<EvaluationResponse> => {
  const formData = new FormData();
  formData.append("file", cvFile); // must match FastAPI param name
  formData.append("job_description", jobDescription);

  const response = await fetch("http://localhost:8000/evaluate", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Evaluation failed: ${response.statusText}`);
  }

  const data: EvaluationResponse = await response.json();
  return data;
};*/

export const evaluateCV = async (cvFile: File, jobDescription: string) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        answer: `CV Analysis for Vector Database Engineer Role
Candidate: Heidi Meek (Wardrobe Stylist)
Target Role: Machine Learning Engineer (Vector Databases & Embeddings)

Key Observations:
Mismatched Profession: The CV is for a wardrobe stylist with extensive experience in fashion, celebrity styling, and advertising. There is no mention of ML, vector databases, or programming.

Zero Overlap with Job Requirements: Skills like ChromaDB, FAISS, embeddings, or ANN search are absent.

Strengths in Creativity/Collaboration: While irrelevant to the role, the CV highlights strong client-facing and project management skills (e.g., coordinating high-profile campaigns).

Strengths (Non-Technical):
- Client/Team Collaboration: Experience working with celebrities and brands suggests adaptability and communication skills.
- Project Coordination: Managed diverse campaigns (editorial, commercials) with tight deadlines.

Weaknesses (For Target Role):
- No Technical Relevance: Missing:
  • Python, FAISS, ChromaDB, or any ML/NLP tools.
  • Embedding models (BERT, OpenAI), RAG pipelines, or ANN optimization.
  • Cloud deployment (AWS/GCP) or large-scale data handling.
- Career Transition Needed: Requires upskilling in ML engineering fundamentals before applying.`
      });
    }, 2000);
  });
};