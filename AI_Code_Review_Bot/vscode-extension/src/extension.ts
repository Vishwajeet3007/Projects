import * as vscode from "vscode";

interface ReviewResponse {
  metadata: Record<string, unknown>;
  report: Record<string, unknown>;
}

function getApiBaseUrl(): string {
  return vscode.workspace.getConfiguration("aiCodeReview").get<string>("apiBaseUrl") || "http://localhost:8000/api/v1";
}

async function postJson(path: string, payload: Record<string, unknown>): Promise<ReviewResponse> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json() as Promise<ReviewResponse>;
}

async function showReviewResult(title: string, result: ReviewResponse): Promise<void> {
  const document = await vscode.workspace.openTextDocument({
    content: JSON.stringify(result, null, 2),
    language: "json",
  });
  await vscode.window.showTextDocument(document, { preview: false });
  void vscode.window.showInformationMessage(`${title} completed.`);
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("aiCodeReview.reviewCurrentFile", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        void vscode.window.showWarningMessage("Open a source file to review.");
        return;
      }

      const document = editor.document;
      try {
        const result = await postJson("/review-code", {
          code: document.getText(),
          filename: document.fileName.split(/[/\\]/).pop(),
          language: document.languageId,
        });
        await showReviewResult("File review", result);
      } catch (error) {
        void vscode.window.showErrorMessage(`Review failed: ${String(error)}`);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("aiCodeReview.reviewRepositoryUrl", async () => {
      const repoUrl = await vscode.window.showInputBox({
        prompt: "Enter the GitHub repository URL",
        placeHolder: "https://github.com/org/repo",
      });
      if (!repoUrl) {
        return;
      }

      try {
        const result = await postJson("/review-repo", { repo_url: repoUrl });
        await showReviewResult("Repository review", result);
      } catch (error) {
        void vscode.window.showErrorMessage(`Repository review failed: ${String(error)}`);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("aiCodeReview.reviewPullRequestUrl", async () => {
      const prUrl = await vscode.window.showInputBox({
        prompt: "Enter the GitHub pull request URL",
        placeHolder: "https://github.com/org/repo/pull/123",
      });
      if (!prUrl) {
        return;
      }

      try {
        const result = await postJson("/review-pr", { pr_url: prUrl });
        await showReviewResult("Pull request review", result);
      } catch (error) {
        void vscode.window.showErrorMessage(`Pull request review failed: ${String(error)}`);
      }
    })
  );
}

export function deactivate(): void {
  // No-op cleanup.
}
