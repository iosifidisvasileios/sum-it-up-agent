from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class PromptEvalCase:
    name: str
    prompt: str
    expected: Dict[str, Any]


DATASET: List[PromptEvalCase] = [
    PromptEvalCase(
        name="summary_only",
        prompt="Summarize this meeting. No email, no pdf.",
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": [],
        },
    ),
    PromptEvalCase(
        name="summary_pdf",
        prompt="Just summarize this, and create a pdf report.",
        expected={
            "wants_summary": True,
            "communication_channels": ["pdf"],
        },
    ),
    PromptEvalCase(
        name="transcription_only",
        prompt="Only transcribe it. Do not summarize.",
        expected={
            "wants_summary": False,
            "wants_transcription": True,
        },
    ),
    PromptEvalCase(
        name="email_with_recipient",
        prompt="Summarize it and email it to bill@example.com with subject Weekly sync.",
        expected={
            "wants_summary": True,
            "communication_channels": ["email"],
            "recipients_contains": ["bill@example.com"],
        },
    ),
    PromptEvalCase(
        name="action_items_and_decisions",
        prompt="Give me action items and decisions, then generate a PDF.",
        expected={
            "wants_summary": True,
            "communication_channels": ["pdf"],
            "summary_types_any": ["action_items", "decisions"],
        },
    ),
    PromptEvalCase(
        name="summary_and_transcription",
        prompt="I want both a transcript and a summary.",
        expected={
            "wants_summary": True,
            "wants_transcription": True,
        },
    ),
    PromptEvalCase(
        name="explicit_no_transcription",
        prompt="Summarize it but do NOT include transcription.",
        expected={
            "wants_summary": True,
            "wants_transcription": False,
        },
    ),
    PromptEvalCase(
        name="email_multiple_recipients",
        prompt="Please summarize and email to alice@example.com and bob@example.com.",
        expected={
            "wants_summary": True,
            "communication_channels": ["email"],
            "recipients_contains": ["alice@example.com", "bob@example.com"],
        },
    ),
    PromptEvalCase(
        name="pdf_and_email",
        prompt="Summarize it, export a PDF, and email it to ops@example.com.",
        expected={
            "wants_summary": True,
            "communication_channels": ["pdf", "email"],
            "recipients_contains": ["ops@example.com"],
        },
    ),
    PromptEvalCase(
        name="executive_summary",
        prompt="Give me an executive summary. No email.",
        expected={
            "wants_summary": True,
            "communication_channels": [],
            "summary_types_any": ["executive"],
        },
    ),
    PromptEvalCase(
        name="bullet_points",
        prompt="Summarize in bullet points and also include key points.",
        expected={
            "wants_summary": True,
            "summary_types_any": ["bullet_points", "key_points"],
        },
    ),
    PromptEvalCase(
        name="custom_instructions_tone",
        prompt="Summarize, but keep it concise and professional. Add a short risks section.",
        expected={
            "wants_summary": True,
            "custom_instructions_contains_any": ["concise", "professional", "risks"],
        },
    ),
    PromptEvalCase(
        name="no_summary_no_transcription",
        prompt="Don't summarize and don't transcribe.",
        expected={
            "wants_summary": False,
            "wants_transcription": False,
        },
    ),
    PromptEvalCase(
        name="complex_customer_followup_multi_channel",
        prompt=(
            "We just finished a 55-minute customer call about the new rollout.\n"
            "Please do the following:\n"
            "1) Create a detailed summary with key points, decisions, and action items.\n"
            "2) Export the summary as a PDF.\n"
            "3) Email the PDF and a short executive recap to customer-success@example.com and pm@example.com.\n"
            "Subject: ACME rollout follow-up - next steps\n"
            "Also: keep the tone professional and concise, and highlight any risks/blockers."
        ),
        expected={
            "wants_summary": True,
            "communication_channels": ["email", "pdf"],
            "recipients_contains": ["customer-success@example.com", "pm@example.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "key_points", "decisions", "action_items", "executive"],
            "custom_instructions_contains_any": ["professional", "concise", "risks", "blockers"],
        },
    ),
    PromptEvalCase(
        name="complex_internal_coordination_slack_jira",
        prompt=(
            "From this meeting, I need a crisp bullet-point summary plus a separate action-items list.\n"
            "Post it to Slack and also create a Jira follow-up.\n"
            "If something is unclear, make reasonable assumptions and add a short 'Open Questions' section."
        ),
        expected={
            "wants_summary": True,
            "communication_channels": ["slack", "jira"],
            "summary_types_any": ["bullet_points", "action_items"],
            "custom_instructions_contains_any": ["open questions", "assumptions"],
        },
    ),
    PromptEvalCase(
        name="complex_discord_telegram_updates",
        prompt=(
            "Summarize this standup with key points and decisions.\n"
            "Send updates to our Discord and Telegram channels.\n"
            "Keep it under 10 bullets and don't include the full transcript."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": ["discord", "telegram"],
            "summary_types_any": ["key_points", "decisions", "bullet_points"],
            "custom_instructions_contains_any": ["under", "bullets"],
        },
    ),
    # New complex test cases with diverse expectations
    PromptEvalCase(
        name="multi_channel_stakeholder_communication",
        prompt=(
            "This was a critical board meeting about Q4 strategy. I need:\n"
            "1) Full transcription for legal records\n"
            "2) Executive summary with financial highlights and risks\n"
            "3) Detailed action items with owners and deadlines\n"
            "4) Email the executive summary to board@company.com and cfo@company.com\n"
            "5) Post action items to the #strategy Slack channel\n"
            "6) Create a PDF report for the audit committee\n"
            "Subject: Q4 Board Meeting - Strategic Decisions & Action Items\n"
            "Keep everything confidential and use formal tone."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": True,
            "communication_channels": ["email", "slack", "pdf"],
            "recipients_contains": ["board@company.com", "cfo@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["executive", "detailed", "action_items", "decisions"],
            "custom_instructions_contains_any": ["confidential", "formal", "financial", "legal"],
        },
    ),
    PromptEvalCase(
        name="customer_support_triage_workflow",
        prompt=(
            "Customer escalation call just ended. Please:\n"
            "- Transcribe the entire conversation for compliance\n"
            "- Extract customer pain points and technical issues\n"
            "- Create a summary with next steps and timeline\n"
            "- Send to support-team@company.com and engineering@company.com\n"
            "- Create Jira tickets for high-priority bugs\n"
            "- Post customer sentiment analysis to #customer-insights Discord\n"
            "- Generate a PDF for the customer success manager\n"
            "Make it actionable and highlight any SLA breaches."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": True,
            "communication_channels": ["email", "jira", "discord", "pdf"],
            "recipients_contains": ["support-team@company.com", "engineering@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "action_items", "key_points"],
            "custom_instructions_contains_any": ["actionable", "sla", "compliance", "sentiment"],
        },
    ),
    PromptEvalCase(
        name="product_launch_coordination",
        prompt=(
            "Product launch sync meeting - need comprehensive coverage:\n"
            "1) Bullet-point summary of launch timeline and milestones\n"
            "2) Marketing decisions and budget allocations\n"
            "3) Technical dependencies and blockers\n"
            "4) Email summary to product@company.com with subject 'Launch Update'\n"
            "5) Post blockers to #dev-team Slack channel\n"
            "6) Create detailed PDF for leadership review\n"
            "7) Send quick recap to marketing-lead@company.com\n"
            "8) Add risk assessment section and mitigation strategies\n"
            "Keep it concise but thorough - no fluff."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": ["email", "slack", "pdf"],
            "recipients_contains": ["product@company.com", "marketing-lead@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["bullet_points", "decisions", "detailed", "action_items"],
            "custom_instructions_contains_any": ["concise", "thorough", "risk", "mitigation"],
        },
    ),
    PromptEvalCase(
        name="crisis_management_response",
        prompt=(
            "URGENT: Security incident response meeting completed.\n"
            "Need immediate action:\n"
            "- Full transcript for incident investigation\n"
            "- Timeline of events and decisions made\n"
            "- Critical action items with owners (security, legal, PR)\n"
            "- Email to incident-response@company.com, legal@company.com, pr@company.com\n"
            "- Post security updates to #alerts Slack and #security Discord\n"
            "- Create confidential PDF for executive team\n"
            "- Generate Jira tickets for all follow-up items\n"
            "Subject: CRITICAL - Security Incident Response Actions\n"
            "Mark everything as high priority and time-sensitive."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": True,
            "communication_channels": ["email", "slack", "discord", "pdf", "jira"],
            "recipients_contains": ["incident-response@company.com", "legal@company.com", "pr@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "action_items", "decisions", "timeline"],
            "custom_instructions_contains_any": ["urgent", "critical", "high priority", "time-sensitive", "confidential"],
        },
    ),
    PromptEvalCase(
        name="research_collaboration_synthesis",
        prompt=(
            "Research collaboration meeting with university partners.\n"
            "Please provide:\n"
            "- Academic-style summary with key findings and methodology\n"
            "- Research questions and hypotheses discussed\n"
            "- Next steps for publication and data sharing\n"
            "- Email to research-team@university.edu and pi@company.com\n"
            "- Create PDF for grant reporting requirements\n"
            "- Post collaboration opportunities to #research Discord\n"
            "- Include references and citations section\n"
            "Use scholarly tone and include IRB considerations."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": ["email", "pdf", "discord"],
            "recipients_contains": ["research-team@university.edu", "pi@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "key_points", "action_items"],
            "custom_instructions_contains_any": ["scholarly", "academic", "publication", "grant", "irb"],
        },
    ),
    PromptEvalCase(
        name="sales_deal_review_multi_format",
        prompt=(
            "Enterprise sales deal review - need comprehensive analysis:\n"
            "1) Transcribe for compliance and record-keeping\n"
            "2) Executive summary with deal value and timeline\n"
            "3) Risk assessment and competitive analysis\n"
            "4) Action items for sales, legal, and technical teams\n"
            "5) Email to sales-leadership@company.com and legal@company.com\n"
            "6) Create detailed PDF for deal review board\n"
            "7) Post key milestones to #sales-wins Slack channel\n"
            "8) Generate CRM follow-up tasks via Jira integration\n"
            "Subject: Enterprise Deal Review - $2.5M Opportunity\n"
            "Highlight any blockers and required approvals."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": True,
            "communication_channels": ["email", "pdf", "slack", "jira"],
            "recipients_contains": ["sales-leadership@company.com", "legal@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["executive", "detailed", "action_items", "decisions"],
            "custom_instructions_contains_any": ["risk", "competitive", "compliance", "blockers", "approvals"],
        },
    ),
    PromptEvalCase(
        name="agile_retrospective_workflow",
        prompt=(
            "Sprint retrospective meeting - process improvements needed:\n"
            "- Bullet points of what went well and what didn't\n"
            "- Action items for process improvements\n"
            "- Team sentiment and engagement feedback\n"
            "- Post to #team-retro Slack channel\n"
            "- Email summary to agile-coach@company.com\n"
            "- Create PDF for sprint review presentation\n"
            "- Generate improvement tickets in Jira\n"
            "- Add celebration section for team achievements\n"
            "Keep it constructive and forward-looking."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": ["slack", "email", "pdf", "jira"],
            "recipients_contains": ["agile-coach@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["bullet_points", "action_items", "key_points"],
            "custom_instructions_contains_any": ["constructive", "forward-looking", "celebration", "improvement"],
        },
    ),
    PromptEvalCase(
        name="compliance_audit_preparation",
        prompt=(
            "SOX compliance audit preparation meeting - critical documentation:\n"
            "1) Full transcription for audit trail\n"
            "2) Detailed summary of control assessments\n"
            "3) Gap analysis and remediation plans\n"
            "4) Action items with compliance owners and deadlines\n"
            "5) Email to compliance@company.com, audit-committee@company.com\n"
            "6) Create comprehensive PDF for external auditors\n"
            "7) Post status updates to #compliance Slack channel\n"
            "8) Generate audit tickets in Jira with proper classification\n"
            "Subject: SOX Audit Preparation - Control Assessments & Gaps\n"
            "Ensure all documentation meets regulatory requirements."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": True,
            "communication_channels": ["email", "pdf", "slack", "jira"],
            "recipients_contains": ["compliance@company.com", "audit-committee@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "action_items", "decisions"],
            "custom_instructions_contains_any": ["sox", "compliance", "audit", "regulatory", "gap"],
        },
    ),
    PromptEvalCase(
        name="cross_functional_project_sync",
        prompt=(
            "Cross-functional project sync - multiple stakeholder alignment:\n"
            "- Project status and milestone achievements\n"
            "- Resource allocation and budget updates\n"
            "- Technical dependencies and integration points\n"
            "- Risk mitigation strategies and contingency plans\n"
            "- Email to project-team@company.com, finance@company.com\n"
            "- Post updates to #project-sync Slack and #project Discord\n"
            "- Create executive PDF for steering committee\n"
            "- Generate resource tickets in Jira\n"
            "Include visual timeline and RACI matrix references."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": ["email", "slack", "discord", "pdf", "jira"],
            "recipients_contains": ["project-team@company.com", "finance@company.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "action_items", "decisions", "key_points"],
            "custom_instructions_contains_any": ["milestone", "resource", "budget", "risk", "raci"],
        },
    ),
]

SYSTEM_PROMPTS: Dict[str, str] = {
    "default": """You are an expert at analyzing user requests for audio processing and summarization tasks.
Extract the user's intent and return a structured JSON response.

Analyze the user's prompt for:
1. Communication channels (email, slack, discord, telegram, jira)
2. Summary types (action_items, decisions, key_points, detailed, bullet_points, executive, standard)
3. Recipients (email addresses or names)
4. Custom instructions (tone, voice/persona, audience, depth, don’ts)

Return a JSON object with this exact structure:
{
    "wants_summary": True/False,
    "wants_transcription": True/False,
    "communication_channels": ["email", "pdf", "slack", "jira", etc.],
    "recipients": ["email@example.com", "Bill_Slack"],
    "subject": "Email subject or null",
    "summary_types": ["action_items", "decisions", etc.],
    "custom_instructions": ["dont use emojis", "be direct", "keep it professional"]
}

Be precise and only include values that are clearly indicated in the prompt.
If something is not mentioned, use null or empty arrays.
Default wants_summary to true unless user explicitly says "transcription only" or "no summary".
Default wants_transcription to False unless user explicitly says "transcription" or "transcribe" or "no summary".
""",
    "strict_json": (
        "You are a parser. Output ONLY valid JSON matching this schema:\n"
        "{\n"
        "  \"wants_summary\": true|false,\n"
        "  \"wants_transcription\": true|false,\n"
        "  \"communication_channels\": [\"email\"|\"pdf\"|\"slack\"|\"jira\"|\"discord\"|\"telegram\"],\n"
        "  \"recipients\": [\"...\"],\n"
        "  \"summary_types\": [\"standard\"|\"action_items\"|\"decisions\"|\"key_points\"|\"detailed\"|\"bullet_points\"|\"executive\"],\n"
        "  \"custom_instructions\": [\"...\"]\n"
        "}\n"
        "Rules: No markdown. No extra keys. Use empty arrays when unknown."
    ),
    "conversational": (
        "You are an intelligent assistant that understands meeting requests. "
        "When given a prompt about what to do with a meeting recording, "
        "please think about what the user wants and provide a structured response.\n\n"
        "Output your response as valid JSON with these fields:\n"
        "- wants_summary: boolean indicating if they want a summary\n"
        "- wants_transcription: boolean indicating if they want transcription\n"
        "- communication_channels: array of output methods (email, pdf, slack, jira, discord, telegram)\n"
        "- recipients: array of email addresses or usernames\n"
        "- summary_types: array of summary formats (standard, action_items, decisions, key_points, detailed, bullet_points, executive)\n"
        "- custom_instructions: array of any specific instructions mentioned\n\n"
        "Be helpful and interpret the user's intent naturally."
    ),
    "step_by_step": (
        "Analyze this meeting request step by step:\n\n"
        "1. Read the user's prompt carefully\n"
        "2. Determine if they want a summary (yes/no)\n"
        "3. Determine if they want transcription (yes/no)\n"
        "4. Identify any communication channels mentioned (email, pdf, slack, jira, discord, telegram)\n"
        "5. Extract any recipients (emails, usernames)\n"
        "6. Identify summary types requested (standard, action_items, decisions, key_points, detailed, bullet_points, executive)\n"
        "7. Note any custom instructions or special requests\n\n"
        "Then output your analysis as valid JSON with the following structure:\n"
        "{\n"
        "  \"wants_summary\": boolean,\n"
        "  \"wants_transcription\": boolean,\n"
        "  \"communication_channels\": array,\n"
        "  \"recipients\": array,\n"
        "  \"summary_types\": array,\n"
        "  \"custom_instructions\": array\n"
        "}\n\n"
        "Be thorough and accurate in your analysis."
    ),
    "minimal_1": "Parse meeting request. Return JSON with: wants_summary (bool), wants_transcription (bool), communication_channels (email/pdf/slack/jira/discord/telegram), recipients (emails/usernames), summary_types (action_items/decisions/key_points/detailed/bullet_points/executive), custom_instructions (strings).",
    "minimal_2": "Extract from meeting prompt: summary wanted?, transcription wanted?, output channels, recipients, summary types, special instructions. Return as JSON object.",
    "minimal_3": "Meeting request parser. Output JSON: wants_summary (true/false), wants_transcription (true/false), communication_channels (array), recipients (array), summary_types (array), custom_instructions (array).",
}

