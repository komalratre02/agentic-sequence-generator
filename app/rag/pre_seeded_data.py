"""
Pre-seeded company knowledge for cloud deployment demos.
This completely bypasses Cloudflare/WAF network blocks by providing
high-quality, pre-scraped text for specific demo URLs.
"""

PRE_SEEDED_DATA = {
    "telekom.com": [
        {"source": "https://www.telekom.com/en/company", "text": "Deutsche Telekom is one of the world's leading integrated telecommunications companies, with over 245 million mobile customers, 25 million fixed-network lines, and 21 million broadband lines. The company provides fixed-network, mobile communications, Internet, and IPTV products for consumers, as well as complex Information and Communication Technology (ICT) solutions for business and corporate customers globally."},
        {"source": "https://www.t-systems.com/", "text": "T-Systems is the enterprise customer unit of Deutsche Telekom. With a footprint in over 20 countries, T-Systems operates IT systems for multinational corporations and public sector institutions. They focus heavily on secure cloud computing, digital supply chains, Internet of Things (IoT), and highly secure enterprise network infrastructures."},
        {"source": "https://www.telekom.com/en/company/digital-transformation", "text": "Magenta AI and Digital Transformation are core pillars of Deutsche Telekom's strategy. By leveraging artificial intelligence, machine learning, and automation, the company is optimizing its network operations, improving customer service through intelligent chatbots, and building new AI-driven product offerings for its B2B clients."},
        {"source": "https://www.telekom.com/en/corporate-responsibility", "text": "Sustainability is deeply embedded in Deutsche Telekom's operations. The company is committed to achieving climate neutrality for its in-house emissions by 2025, and completely across its entire value chain by 2040. They actively promote digital inclusion, ethical AI, and green energy solutions across their data centers and cell towers."}
    ],
    "stripe.com": [
        {"source": "https://stripe.com/", "text": "Stripe is a financial infrastructure platform for the internet. Millions of companies—from the world's largest enterprises to the most ambitious startups—use Stripe to accept payments, grow their revenue, and accelerate new business opportunities. Headquartered in San Francisco and Dublin, the company aims to increase the GDP of the internet."},
        {"source": "https://stripe.com/payments", "text": "Stripe Payments is a fully integrated suite of payments products. We bring together everything that's required to build websites and apps that accept payments and send payouts globally. Stripe's products power payments for online and in-person retailers, subscriptions businesses, software platforms and marketplaces, and everything in between."},
        {"source": "https://stripe.com/billing", "text": "Stripe Billing is the fastest way to build and scale recurring revenue. You can implement billing models like flat-rate, per-seat, or usage-based pricing in minutes. It automatically handles prorations, invoicing, and tax collection, significantly reducing the engineering overhead required to manage subscriptions."},
        {"source": "https://stripe.com/connect", "text": "Stripe Connect is the fastest and easiest way to integrate payments into your software platform or marketplace. Connect handles the complexities of global routing, onboarding, and compliance so you can focus on building your core product. It supports multiple business models including standard, express, and custom accounts."},
        {"source": "https://stripe.com/radar", "text": "Stripe Radar provides machine learning fraud protection integrated directly into Stripe Payments. It helps detect and block fraud using data across the entire Stripe network. Radar is trained on hundreds of billions of data points to optimize authorization rates while preventing fraudulent chargebacks."},
        {"source": "https://stripe.com/atlas", "text": "Stripe Atlas is a powerful, safe, and easy-to-use platform for forming a company. By removing lengthy paperwork, legal complexity, and numerous fees, Stripe Atlas helps you launch your startup from anywhere in the world. It includes C-Corporation formation in Delaware, an IRS Employer Identification Number (EIN), and a U.S. bank account."}
    ],
    "linear.app": [
        {"source": "https://linear.app/", "text": "Linear is a purpose-built tool for software product development. It streamlines issues, sprints, and product roadmaps so teams can do their best work. Linear is designed to be ridiculously fast, with keyboard-first navigation and an incredibly responsive interface that gets out of your way."},
        {"source": "https://linear.app/method", "text": "The Linear Method outlines our principles for building software. We believe in opinionated software that guides you towards better workflows. Instead of infinitely configurable setups that become chaotic, Linear enforces simple, effective structures like Cycles (instead of Sprints) to maintain momentum."},
        {"source": "https://linear.app/features/issues", "text": "Issue tracking in Linear is built for speed. Create issues in seconds, assign them to team members, and move them through customized workflows. Keyboard shortcuts let you manage triage, assign labels, and update status without ever touching your mouse. Everything syncs instantly in real-time."},
        {"source": "https://linear.app/features/projects", "text": "Linear Projects help you plan and execute larger initiatives. Group related issues, set target dates, and track progress automatically via project milestones. Projects give leadership clear visibility into what's shipping next without requiring engineers to manually update spreadsheets or Gantt charts."}
    ],
    "posthog.com": [
        {"source": "https://posthog.com/", "text": "PostHog is the open-source product OS. It provides product analytics, session recording, feature flags, A/B testing, and surveys all in one platform. Built for engineers, PostHog enables teams to build better products without needing to wire together six different SaaS tools."},
        {"source": "https://posthog.com/product-analytics", "text": "PostHog Product Analytics helps you understand user behavior. Create funnels to identify where users drop off, use trends to track core metrics over time, and build retention cohorts. Because everything is tracked via autocapture by default, you don't need to manually instrument events before answering product questions."},
        {"source": "https://posthog.com/session-replay", "text": "Session Replay lets you watch exactly how users experience your app. It records DOM mutations so you can playback sessions perfectly, identify UX issues, and debug console errors instantly. It's fully integrated with analytics—if you see a drop-off in a funnel, you can click to watch the exact sessions of users who failed to convert."},
        {"source": "https://posthog.com/feature-flags", "text": "PostHog Feature Flags allow you to safely roll out new features. Toggle functionality on or off without deploying code, roll out to specific percentages of users, or target specific user properties. Because it's integrated with analytics, you can automatically track the impact of a feature flag on your core conversion metrics."}
    ]
}

def get_pre_seeded_chunks(url: str) -> list[dict]:
    """Return pre-seeded chunks if the URL matches a demo domain."""
    url_lower = url.lower()
    for domain, chunks in PRE_SEEDED_DATA.items():
        if domain in url_lower:
            return chunks
    return []
