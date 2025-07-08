import { useState } from "react";
import RecruiterForm from "@/pages/RecruiterForm";
import CVUpload from "@/pages/CVUpload";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const Dashboard = () => {
  const [selectedRole, setSelectedRole] = useState<null | "recruiter" | "candidate">(null);
  const [loggedInRole, setLoggedInRole] = useState<null | "recruiter" | "candidate">(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [score, setScore] = useState<number | null>(null);
  const [feedback, setFeedback] = useState("");

  const handleLogin = () => {
    if (!username || !password) {
      alert("Please enter username and password.");
      return;
    }

    // Optional: Add actual validation logic here (e.g., check hardcoded credentials or call backend)
    setLoggedInRole(selectedRole);
  };

  const handleLogout = () => {
    setLoggedInRole(null);
    setSelectedRole(null);
    setUsername("");
    setPassword("");
    setScore(null);
    setFeedback("");
  };

  const handleResult = (newScore: number, newFeedback: string) => {
    setScore(newScore);
    setFeedback(newFeedback);
  };

  if (!loggedInRole) {
    return (
      <div className="p-6 max-w-md mx-auto space-y-6 text-center">
        <h1 className="text-2xl font-bold mb-4">CV Align - Dashboard Login</h1>

        {!selectedRole ? (
          <div className="space-y-4">
            <Button onClick={() => setSelectedRole("recruiter")}>Login as Recruiter</Button>
            <Button onClick={() => setSelectedRole("candidate")}>Login as Candidate</Button>
          </div>
        ) : (
          <div className="space-y-4 text-left">
            <p className="text-lg font-medium text-center">
              {selectedRole === "recruiter" ? "Recruiter" : "Candidate"} Login
            </p>
            <Input
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
            <Input
              placeholder="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <div className="flex gap-2 justify-center">
              <Button onClick={handleLogin}>Login</Button>
              <Button variant="outline" onClick={() => setSelectedRole(null)}>
                Back
              </Button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">
          CV Align - {loggedInRole === "recruiter" ? "Recruiter" : "Candidate"} Portal
        </h1>
        <Button variant="outline" onClick={handleLogout}>
          Logout
        </Button>
      </div>

      {loggedInRole === "recruiter" && <RecruiterForm onResult={handleResult} />}
      {loggedInRole === "candidate" && <CVUpload onResult={handleResult} />}

      <Card className="mt-6">
        <CardContent>
          {feedback != null ? (
            <>
              <p className="text-lg font-semibold">{score}</p>
              <p className="whitespace-pre-wrap">{feedback}</p>
            </>
          ) : (
            <p>No results yet. Submit your form to evaluate.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;
