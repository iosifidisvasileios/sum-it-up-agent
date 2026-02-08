from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Type


class PromptTemplate(ABC):
    """Interface for a meeting summarization prompt template."""
    meeting_type: str

    @abstractmethod
    def render(self, transcript: str) -> str:
        """Return the full prompt string."""
        raise NotImplementedError


# -------------------------
# Concrete template classes
# -------------------------

@dataclass(frozen=True)
class TeamStatusSyncStandupTemplate(PromptTemplate):
    meeting_type: str = "team status sync / standup"

    def render(self, transcript: str) -> str:
        return f"""\
You are an expert meeting scribe. Summarize the conversation as a TEAM STATUS SYNC / STANDUP.

INPUT
- You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
- The transcript may contain filler words, interruptions, and incomplete sentences.

GOALS
- Produce a concise, high-signal standup summary that highlights: progress, blockers, next steps, and asks.
- Preserve accountability (who owns what) and any concrete dates or metrics.
- Do not invent details. If an owner or deadline is unclear, mark it as "Owner: Unknown" / "Due: Unknown".

OUTPUT FORMAT (STRICT)
Return valid JSON only (no markdown), using exactly this schema:

{{
  "meeting_type": "team status sync / standup",
  "title": "<short descriptive title>",
  "time_range": {{"start_sec": <float>, "end_sec": <float>}},
  "participants": ["SPEAKER_00", "SPEAKER_01", "..."],
  "executive_summary": "<3-6 sentences, plain English>",
  "updates_by_speaker": [
    {{
      "speaker": "SPEAKER_00",
      "updates": ["<what they did / progress>", "..."],
      "blockers": ["<blocker>", "..."],
      "asks": ["<ask/request>", "..."],
      "next_steps": ["<next step>", "..."]
    }}
  ],
  "team_blockers": ["<cross-cutting blockers>", "..."],
  "decisions": [
    {{
      "decision": "<what was decided>",
      "owner": "<speaker or Unknown>",
      "timestamp_sec": <float or null>
    }}
  ],
  "action_items": [
    {{
      "owner": "<speaker or Unknown>",
      "task": "<clear task statement>",
      "due": "<ISO date if stated, else Unknown>",
      "timestamp_sec": <float or null>
    }}
  ],
  "risks": ["<risk statement>", "..."],
  "open_questions": ["<question needing follow-up>", "..."],
  "tags": ["standup", "status", "<domain tag>", "..."]
}}

RULES
- Be brief: prefer bullets in arrays; avoid long paragraphs.
- Extract dates exactly as said; if relative ("tomorrow"), keep it as spoken and also add a best-effort ISO date only if the transcript includes an absolute meeting date; otherwise keep "Unknown".
- Include an item in updates_by_speaker only if that speaker contributed substantive content.
- If no decisions/action_items exist, return empty arrays.
- time_range should reflect the transcript boundaries (first/last timestamp observed).

TRANSCRIPT
{transcript}
"""


# -------------------------
# Placeholders for others
# -------------------------

@dataclass(frozen=True)
class PlanningCoordinationMeetingTemplate(PromptTemplate):
    meeting_type: str = "planning / coordination meeting"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert program manager and meeting scribe. Summarize the conversation as a PLANNING / COORDINATION MEETING.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include brainstorming, partial ideas, and unresolved items.

    GOALS
    - Produce a planning-grade summary focused on: scope, plan, owners, dependencies, timelines, and next steps.
    - Extract explicit commitments and convert them into actionable tasks with clear owners and due dates when stated.
    - Identify dependencies, risks, and open questions that could block execution.
    - Do not invent details. If an owner or date is unclear, use "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "planning / coordination meeting",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "objective": "<1-2 sentences describing the primary goal of the meeting>",
      "executive_summary": "<3-7 sentences capturing the plan and coordination outcomes>",

      "plan_overview": [
        "<ordered step or phase 1>",
        "<ordered step or phase 2>",
        "..."
      ],

      "workstreams": [
        {{
          "name": "<workstream name, inferred from transcript>",
          "owner": "<speaker or Unknown>",
          "scope": ["<in scope item>", "..."],
          "out_of_scope": ["<explicitly out of scope item>", "..."],
          "dependencies": ["<dependency>", "..."],
          "milestones": [
            {{
              "milestone": "<milestone description>",
              "due": "<ISO date if stated, else Unknown>",
              "timestamp_sec": <float or null>
            }}
          ],
          "risks": ["<risk>", "..."],
          "open_questions": ["<question>", "..."]
        }}
      ],

      "decisions": [
        {{
          "decision": "<what was decided>",
          "rationale": "<why, if stated>",
          "owner": "<speaker or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<clear task statement>",
          "due": "<ISO date if stated, else Unknown>",
          "dependency": "<dependency if any, else null>",
          "timestamp_sec": <float or null>
        }}
      ],

      "assumptions": ["<assumption>", "..."],
      "risks": ["<cross-cutting risk>", "..."],
      "open_questions": ["<question needing follow-up>", "..."],
      "next_meeting": {{
        "needed": <true/false>,
        "proposed_agenda": ["<agenda item>", "..."],
        "proposed_time": "<as stated, else Unknown>"
      }},
      "tags": ["planning", "coordination", "<domain tag>", "..."]
    }}

    RULES
    - Be execution-oriented. Prefer short bullets in arrays over long prose.
    - Convert “we should / let’s / need to” statements into action_items when they imply work.
    - If a due date is relative (“tomorrow”), keep it as spoken and only add an ISO date if an absolute meeting date is present in the transcript; otherwise use "Unknown".
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class DecisionMakingMeetingTemplate(PromptTemplate):
    meeting_type: str = "decision-making meeting"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert facilitator and meeting scribe. Summarize the conversation as a DECISION-MAKING MEETING.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may contain debate, uncertainty, and partial proposals.

    GOALS
    - Identify the core decision(s) the group is trying to make.
    - Capture options considered, evaluation criteria, trade-offs, and final decision(s).
    - Record dissent/risks and any follow-up actions required to implement decisions.
    - Do not invent details. If a decision or owner is unclear, mark it as "Unknown" and reflect uncertainty.

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "decision-making meeting",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "decision_summary": "<3-7 sentences summarizing what was decided and why>",

      "decisions": [
        {{
          "decision": "<final decision statement>",
          "status": "<decided | tentative | deferred>",
          "owner": "<speaker or Unknown>",
          "timestamp_sec": <float or null>,
          "rationale": ["<reason>", "..."],
          "criteria": ["<criterion used to choose>", "..."],
          "tradeoffs": ["<trade-off acknowledged>", "..."],
          "dissent_or_concerns": ["<concern or dissent>", "..."],
          "follow_ups_required": <true/false>
        }}
      ],

      "options_considered": [
        {{
          "topic": "<what the decision is about>",
          "options": [
            {{
              "name": "<option name/label>",
              "pros": ["<pro>", "..."],
              "cons": ["<con>", "..."],
              "risks": ["<risk>", "..."],
              "cost_or_effort": "<low|medium|high|Unknown>",
              "impact": "<low|medium|high|Unknown>"
            }}
          ],
          "recommended_option": "<name or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "assumptions": ["<assumption>", "..."],
      "risks": ["<risk>", "..."],
      "open_questions": ["<open question>", "..."],

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<implementation or validation step>",
          "due": "<ISO date if stated, else Unknown>",
          "related_decision": "<short reference to decision>",
          "timestamp_sec": <float or null>
        }}
      ],

      "parking_lot": ["<topic explicitly parked for later>", "..."],
      "tags": ["decision", "tradeoffs", "<domain tag>", "..."]
    }}

    RULES
    - Be explicit: decisions must be crisp, testable statements (“We will … by …”).
    - Distinguish decided vs tentative vs deferred; do not overstate certainty.
    - If multiple decisions exist, list them separately.
    - Extract criteria/trade-offs only if stated or clearly implied by discussion (e.g., “too risky”, “cheaper”, “faster”).
    - If no decisions are actually made, set decisions=[] and explain in decision_summary that outcomes were deferred.
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class BrainstormingSessionTemplate(PromptTemplate):
    meeting_type: str = "brainstorming session"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert facilitator and product strategist. Summarize the conversation as a BRAINSTORMING SESSION.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include half-formed ideas, tangents, and speculative discussion.

    GOALS
    - Capture the idea space clearly without overcommitting to outcomes.
    - Group similar ideas into themes, and preserve notable raw ideas verbatim-ish (paraphrase, no long quotes).
    - Extract evaluation criteria, constraints, and assumptions raised during ideation.
    - Identify top candidates (if any) and define next steps to validate them.
    - Do not invent details. If priority/owner is unclear, use "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "brainstorming session",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "problem_statement": "<1-2 sentences: what are we trying to solve?>",
      "executive_summary": "<3-7 sentences summarizing the ideation outcomes>",

      "themes": [
        {{
          "theme": "<theme name>",
          "summary": "<1-3 sentences summarizing the theme>",
          "ideas": [
            {{
              "idea": "<idea statement>",
              "proposed_by": "<speaker or Unknown>",
              "timestamp_sec": <float or null>,
              "notes": ["<detail/variant>", "..."],
              "assumptions": ["<assumption>", "..."],
              "dependencies": ["<dependency>", "..."],
              "risks": ["<risk>", "..."],
              "effort": "<low|medium|high|Unknown>",
              "impact": "<low|medium|high|Unknown>"
            }}
          ]
        }}
      ],

      "wildcards": [
        {{
          "idea": "<unusual/left-field idea worth noting>",
          "proposed_by": "<speaker or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "evaluation_criteria": ["<criterion used to judge ideas>", "..."],
      "constraints": ["<constraint: budget/time/tech/legal/etc>", "..."],

      "top_candidates": [
        {{
          "idea": "<idea name/short label>",
          "why_promising": ["<reason>", "..."],
          "open_questions": ["<question>", "..."],
          "next_validation_steps": ["<step>", "..."],
          "owner": "<speaker or Unknown>"
        }}
      ],

      "decisions": [
        {{
          "decision": "<only if a decision was actually made>",
          "owner": "<speaker or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<validation task / research / prototype>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "open_questions": ["<unresolved question>", "..."],
      "tags": ["brainstorming", "ideas", "<domain tag>", "..."]
    }}

    RULES
    - Do not “decide” for the group. Only fill decisions if the transcript includes explicit agreement.
    - Prefer grouping over listing; reduce duplicates by consolidating similar ideas under one theme.
    - Keep ideas concrete and testable (“Prototype X”, “Try Y with audience Z”), not vague.
    - If effort/impact are not stated, set to "Unknown".
    - If no top candidates were identified, return top_candidates=[].
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class RetrospectivePostmortemTemplate(PromptTemplate):
    meeting_type: str = "retrospective / postmortem"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an experienced incident/postmortem facilitator and retrospective scribe. Summarize the conversation as a RETROSPECTIVE / POSTMORTEM.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include incomplete timelines, hypotheses, and differing perspectives.

    GOALS
    - Reconstruct what happened: timeline, impact, detection, response, resolution, and follow-ups.
    - Identify contributing factors and root cause(s) only when supported; otherwise label as "hypothesis".
    - Capture what went well, what didn’t, and concrete preventive actions with owners.
    - Maintain a blameless tone.
    - Do not invent details.

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "retrospective / postmortem",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "incident_or_topic": "<what this retro/postmortem is about>",
      "executive_summary": "<4-10 sentences covering impact, cause, and key learnings>",

      "impact": {{
        "summary": "<who/what was affected and how>",
        "severity": "<sev0|sev1|sev2|sev3|Unknown>",
        "start_time": "<as stated, else Unknown>",
        "end_time": "<as stated, else Unknown>",
        "user_impact": ["<impact detail>", "..."],
        "business_impact": ["<impact detail>", "..."],
        "metrics": ["<metric changes if mentioned>", "..."]
      }},

      "timeline": [
        {{
          "time": "<absolute or relative time as stated, else Unknown>",
          "event": "<what happened>",
          "owner": "<speaker/team or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "detection_and_response": {{
        "detection": ["<how it was detected>", "..."],
        "response_actions": ["<what was done during incident>", "..."],
        "communication": ["<customer/internal comms>", "..."],
        "tools_or_systems": ["<systems involved>", "..."]
      }},

      "root_cause": {{
        "confirmed": ["<confirmed root cause>", "..."],
        "hypotheses": ["<unconfirmed plausible cause>", "..."],
        "contributing_factors": ["<factor>", "..."]
      }},

      "what_went_well": ["<item>", "..."],
      "what_didnt_go_well": ["<item>", "..."],
      "where_we_got_lucky": ["<item>", "..."],

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<preventive/corrective action>",
          "type": "<prevention|detection|process|documentation|testing|monitoring|rollback|other>",
          "priority": "<P0|P1|P2|Unknown>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "follow_up_questions": ["<question>", "..."],
      "tags": ["retro", "postmortem", "learning", "<domain tag>", "..."]
    }}

    RULES
    - Blameless language; focus on systems/processes.
    - Do not claim a root cause unless the transcript indicates confirmation. Otherwise place it under hypotheses.
    - If you cannot infer severity, use "Unknown".
    - Extract times exactly as stated; do not fabricate absolute timestamps.
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class TrainingOnboardingTemplate(PromptTemplate):
    meeting_type: str = "training / onboarding"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert instructor and onboarding scribe. Summarize the conversation as a TRAINING / ONBOARDING SESSION.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include questions, clarifications, step-by-step explanations, and examples.

    GOALS
    - Produce a structured learning summary: objectives, key concepts, procedures, examples, and resources.
    - Capture explicit instructions and “how-to” steps accurately.
    - Extract FAQs (questions asked + answers) and common pitfalls.
    - Identify follow-up tasks for the trainee(s) and any access/tools they need.
    - Do not invent details. If something is unclear, mark as "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "training / onboarding",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "training_goal": "<1-2 sentences: what the session aimed to teach/cover>",
      "audience_level": "<new hire|beginner|intermediate|advanced|Unknown>",
      "executive_summary": "<4-10 sentences summarizing what was taught and outcomes>",

      "learning_objectives": ["<objective>", "..."],

      "key_concepts": [
        {{
          "concept": "<concept name>",
          "explanation": "<2-5 sentences>",
          "why_it_matters": "<1-2 sentences or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "procedures": [
        {{
          "name": "<procedure name>",
          "steps": ["<step 1>", "<step 2>", "..."],
          "inputs": ["<input>", "..."],
          "outputs": ["<output>", "..."],
          "tools_or_systems": ["<tool/system>", "..."],
          "common_pitfalls": ["<pitfall>", "..."],
          "timestamp_sec": <float or null>
        }}
      ],

      "examples": [
        {{
          "example": "<brief description of example/demo>",
          "takeaways": ["<takeaway>", "..."],
          "timestamp_sec": <float or null>
        }}
      ],

      "qa": [
        {{
          "question": "<question asked>",
          "asked_by": "<speaker or Unknown>",
          "answer": "<answer as stated (paraphrased)>",
          "answered_by": "<speaker or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "resources": [
        {{
          "name": "<doc/link/tool name>",
          "description": "<what it is for>",
          "owner": "<speaker or Unknown>"
        }}
      ],

      "access_or_setup_needed": [
        {{
          "item": "<access/tool/setup>",
          "owner": "<speaker or Unknown>",
          "status": "<needed|requested|granted|Unknown>"
        }}
      ],

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<practice task / onboarding to-do>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "open_questions": ["<unresolved question>", "..."],
      "tags": ["training", "onboarding", "how-to", "<domain tag>", "..."]
    }}

    RULES
    - Prioritize clarity and teachability.
    - Convert “you should / you’ll need to / make sure to” into procedures or action_items when appropriate.
    - Do not add external recommendations not present in transcript.
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class InterviewTemplate(PromptTemplate):
    meeting_type: str = "interview"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert interviewer and interview note-taker. Summarize the conversation as an INTERVIEW.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include incomplete sentences, tangents, and follow-up questions.

    GOALS
    - Capture the role/context, the questions asked, and the candidate’s answers.
    - Extract evidence of skills/competencies, achievements, and concrete examples.
    - Record any decisions/outcomes (next round, take-home, rejection, etc.) if stated.
    - Do not invent details. If something is unclear, mark it as "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "interview",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "context": {{
        "role": "<role being interviewed for, if stated, else Unknown>",
        "company_or_team": "<company/team, if stated, else Unknown>",
        "stage": "<screen|technical|onsite|behavioral|hiring manager|other|Unknown>",
        "format": "<phone|video|in-person|Unknown>"
      }},

      "executive_summary": "<4-10 sentences summarizing the interview content and outcomes>",

      "questions_and_answers": [
        {{
          "question": "<question asked>",
          "asked_by": "<speaker or Unknown>",
          "candidate_answer": "<concise paraphrase of the answer>",
          "answered_by": "<speaker (candidate) or Unknown>",
          "follow_ups": ["<follow-up question>", "..."],
          "evidence": ["<concrete evidence: metric/result/tech used>", "..."],
          "timestamp_sec": <float or null>
        }}
      ],

      "signals": {{
        "strengths": ["<strength supported by transcript>", "..."],
        "concerns": ["<concern supported by transcript>", "..."],
        "unknowns": ["<area not covered / unclear>", "..."]
      }},

      "notable_projects_or_examples": [
        {{
          "summary": "<project/example>",
          "impact": ["<impact>", "..."],
          "tools_or_tech": ["<tech>", "..."],
          "timestamp_sec": <float or null>
        }}
      ],

      "candidate_questions": [
        {{
          "question": "<question asked by candidate>",
          "answer": "<answer given>",
          "timestamp_sec": <float or null>
        }}
      ],

      "outcome": {{
        "status": "<advance|reject|pending|Unknown>",
        "next_steps": ["<next step>", "..."],
        "deadline": "<as stated, else Unknown>"
      }},

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<follow-up task (send take-home, schedule round, share materials)>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "tags": ["interview", "<domain tag>", "..."]
    }}

    RULES
    - Use neutral language; do not add subjective judgments not grounded in transcript.
    - Keep answers concise; prefer extracting measurable evidence.
    - If the candidate is not clearly identifiable, treat speakers neutrally (e.g., “SPEAKER_01 (candidate?)” in evidence/notes only when implied).
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class CustomerCallSalesDemoTemplate(PromptTemplate):
    meeting_type: str = "customer call / sales demo"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert sales engineer and customer meeting scribe. Summarize the conversation as a CUSTOMER CALL / SALES DEMO.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include product names, feature discussion, objections, pricing questions, and next steps.

    GOALS
    - Capture customer context, needs, pain points, requirements, objections, and success criteria.
    - Record what was demonstrated (features/workflows) and customer reactions.
    - Extract commitments: next steps, owners, timelines, and requested follow-ups.
    - Do not invent details. If a detail is unclear, use "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "customer call / sales demo",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "account_context": {{
        "customer_name": "<if stated, else Unknown>",
        "industry": "<if stated, else Unknown>",
        "customer_roles": ["<role/title if stated>", "..."],
        "use_case": "<primary use case>",
        "current_solution": "<current tools/process, if stated, else Unknown>"
      }},

      "executive_summary": "<4-10 sentences summarizing needs, demo highlights, and outcomes>",

      "customer_needs": {{
        "pain_points": ["<pain point>", "..."],
        "requirements": ["<requirement>", "..."],
        "constraints": ["<constraint: budget, security, timeline, legal>", "..."],
        "success_criteria": ["<how they measure success>", "..."]
      }},

      "demo_coverage": [
        {{
          "feature_or_workflow": "<what was shown/discussed>",
          "customer_reaction": "<positive|neutral|negative|Unknown>",
          "notes": ["<detail>", "..."],
          "timestamp_sec": <float or null>
        }}
      ],

      "questions_and_objections": [
        {{
          "topic": "<pricing|security|integration|performance|support|implementation|other>",
          "question_or_objection": "<customer question/objection>",
          "response": "<how the team responded>",
          "owner": "<speaker/team or Unknown>",
          "timestamp_sec": <float or null>,
          "status": "<answered|needs_follow_up|Unknown>"
        }}
      ],

      "competitive_or_alternatives": [
        {{
          "name": "<competitor/alternative mentioned>",
          "context": "<why it came up>",
          "timestamp_sec": <float or null>
        }}
      ],

      "commercials": {{
        "pricing_discussed": <true/false>,
        "budget_range": "<if stated, else Unknown>",
        "procurement_process": "<if stated, else Unknown>",
        "security_or_legal": ["<requirement>", "..."]
      }},

      "decisions": [
        {{
          "decision": "<decision made (e.g., proceed to trial)>",
          "owner": "<speaker/team or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "next_steps": [
        {{
          "owner": "<speaker/team or Unknown>",
          "task": "<clear next step>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "risks": ["<risk to deal/progress>", "..."],
      "open_questions": ["<unanswered question>", "..."],
      "tags": ["customer", "sales", "demo", "<domain tag>", "..."]
    }}

    RULES
    - Be customer-centric: prioritize needs/requirements over internal chatter.
    - Do not fabricate customer/company names; if not stated, use "Unknown".
    - If pricing is mentioned vaguely, do not infer numbers.
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class SupportIncidentCallTemplate(PromptTemplate):
    meeting_type: str = "support / incident call"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an experienced on-call incident commander and scribe. Summarize the conversation as a SUPPORT / INCIDENT CALL.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may include hypotheses, rapid updates, commands, and partial timelines.

    GOALS
    - Capture: incident description, scope/impact, detection, suspected causes, mitigations, current status, and next actions.
    - Separate confirmed facts from hypotheses.
    - Extract clear action items with owners and urgency.
    - Do not invent details. If a detail is unclear, use "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "support / incident call",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "incident": {{
        "summary": "<what is broken/what is happening>",
        "status": "<investigating|identified|mitigating|monitoring|resolved|Unknown>",
        "severity": "<sev0|sev1|sev2|sev3|Unknown>",
        "started_at": "<as stated, else Unknown>",
        "detected_at": "<as stated, else Unknown>",
        "affected_services": ["<service/system>", "..."],
        "affected_users": ["<who is impacted>", "..."],
        "symptoms": ["<symptom>", "..."],
        "metrics_signals": ["<error rate/latency/etc if mentioned>", "..."]
      }},

      "executive_summary": "<4-10 sentences summarizing what happened and what is being done>",

      "timeline": [
        {{
          "time": "<absolute or relative time as stated, else Unknown>",
          "event": "<key event/update>",
          "owner": "<speaker/team or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "observations_confirmed": ["<confirmed fact>", "..."],
      "hypotheses": ["<unconfirmed suspected cause>", "..."],

      "mitigations_and_changes": [
        {{
          "action": "<mitigation/change applied or proposed>",
          "type": "<rollback|restart|config_change|feature_flag|traffic_shift|hotfix|other>",
          "status": "<planned|in_progress|done|Unknown>",
          "owner": "<speaker/team or Unknown>",
          "timestamp_sec": <float or null>,
          "result": "<observed effect if mentioned, else Unknown>"
        }}
      ],

      "customer_communications": [
        {{
          "audience": "<internal|external|customer|Unknown>",
          "message": "<what was communicated>",
          "owner": "<speaker/team or Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "risks": ["<risk>", "..."],
      "open_questions": ["<question>", "..."],

      "action_items": [
        {{
          "owner": "<speaker/team or Unknown>",
          "task": "<clear next action>",
          "urgency": "<immediate|today|this_week|Unknown>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "handoffs": [
        {{
          "from": "<speaker/team or Unknown>",
          "to": "<speaker/team or Unknown>",
          "what": "<responsibility handed off>",
          "timestamp_sec": <float or null>
        }}
      ],

      "next_update": {{
        "when": "<as stated, else Unknown>",
        "channel": "<slack/bridge/email/etc if stated, else Unknown>"
      }},

      "tags": ["incident", "support", "oncall", "<domain tag>", "..."]
    }}

    RULES
    - Mark facts vs hypotheses explicitly; do not treat guesses as confirmed.
    - Prefer short, operational language.
    - If incident status/severity are not stated, use "Unknown".
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

@dataclass(frozen=True)
class OtherTemplate(PromptTemplate):
    meeting_type: str = "other"

    def render(self, transcript: str) -> str:
        return f"""\
    You are an expert meeting scribe. The conversation type is UNKNOWN/OTHER.
    Your job is to produce a faithful, general-purpose summary without assuming a specific meeting format.

    INPUT
    - You will receive a transcript with timestamps and speaker labels (e.g., SPEAKER_00).
    - The transcript may be messy, informal, or multi-topic.

    GOALS
    - Provide a clear overview of what was discussed, grouped by topic.
    - Extract decisions, action items, and open questions where they exist.
    - Highlight any commitments, deadlines, owners, and key facts.
    - Do not invent details. If an owner or date is unclear, use "Unknown".

    OUTPUT FORMAT (STRICT)
    Return valid JSON only (no markdown), using exactly this schema:

    {{
      "meeting_type": "other",
      "title": "<short descriptive title>",
      "time_range": {{"start_sec": <float>, "end_sec": <float>}},
      "participants": ["SPEAKER_00", "SPEAKER_01", "..."],

      "executive_summary": "<4-10 sentences summarizing the conversation at a high level>",

      "topics": [
        {{
          "topic": "<topic name>",
          "summary": "<2-6 sentences>",
          "key_points": ["<point>", "..."],
          "notable_quotes_short": ["<very short paraphrase of a notable line>", "..."],
          "timestamp_sec": <float or null>
        }}
      ],

      "decisions": [
        {{
          "decision": "<what was decided>",
          "owner": "<speaker or Unknown>",
          "timestamp_sec": <float or null
          >
        }}
      ],

      "action_items": [
        {{
          "owner": "<speaker or Unknown>",
          "task": "<clear task statement>",
          "due": "<ISO date if stated, else Unknown>",
          "timestamp_sec": <float or null>
        }}
      ],

      "open_questions": ["<question needing follow-up>", "..."],
      "risks": ["<risk or concern>", "..."],

      "classification_guess": {{
        "likely_type": "<one of: team status sync / standup | planning / coordination meeting | decision-making meeting | brainstorming session | retrospective / postmortem | training / onboarding | interview | customer call / sales demo | support / incident call | other>",
        "confidence": "<low|medium|high>",
        "evidence": ["<short evidence snippet/paraphrase from transcript>", "..."]
      }},

      "tags": ["general", "<domain tag>", "..."]
    }}

    RULES
    - Keep topics grouped; avoid duplicating the same point across multiple topics.
    - Do not force a meeting type; classification_guess is a best-effort and may remain "other" with low confidence.
    - time_range must reflect transcript boundaries (first/last timestamp observed).
    - Include only items supported by the transcript; never hallucinate.

    TRANSCRIPT
    {transcript}
    """

# -------------------------
# Factory
# -------------------------

class PromptTemplateFactory:
    _registry: Dict[str, Type[PromptTemplate]] = {}

    @classmethod
    def register(cls, template_cls: Type[PromptTemplate]) -> None:
        meeting_type = getattr(template_cls, "meeting_type", None)
        if not meeting_type:
            raise ValueError(f"{template_cls.__name__} must define meeting_type")
        cls._registry[meeting_type] = template_cls

    @classmethod
    def create(cls, meeting_type: str) -> PromptTemplate:
        if meeting_type not in cls._registry:
            raise KeyError(
                f"Unknown meeting_type: {meeting_type!r}. "
                f"Available: {', '.join(sorted(cls._registry))}"
            )
        return cls._registry[meeting_type]()

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._registry.keys())


# Register defaults (call once at import time)
PromptTemplateFactory.register(TeamStatusSyncStandupTemplate)
PromptTemplateFactory.register(PlanningCoordinationMeetingTemplate)
PromptTemplateFactory.register(DecisionMakingMeetingTemplate)
PromptTemplateFactory.register(BrainstormingSessionTemplate)
PromptTemplateFactory.register(RetrospectivePostmortemTemplate)
PromptTemplateFactory.register(TrainingOnboardingTemplate)
PromptTemplateFactory.register(InterviewTemplate)
PromptTemplateFactory.register(CustomerCallSalesDemoTemplate)
PromptTemplateFactory.register(SupportIncidentCallTemplate)
PromptTemplateFactory.register(OtherTemplate)

# Example:
# tpl = PromptTemplateFactory.create("team status sync / standup")
# prompt = tpl.render(transcript_text)
