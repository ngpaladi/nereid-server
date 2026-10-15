# Nereid and AI-Generated Code

## AI-Generated Code Acknowledgement
We would like to be upfront and acknowledge that we **do** use AI-generated code as part of Nereid. We would also like to be clear as to how we use AI tools to accelerate development:

 - We use AI tools to assist with generating documentation, pull request summaries/reviews, and setting up Github Actions.
 - We ensure all software architecture decisions are made by human developers and maintainers.
 - We do not give short, general prompts for feature implementation. Instead, we give clear, detailed instructions and treat agents as a faster way to type and test.
 - We use Rust for its memory safety features and to minimize potential for security issues, though this project is more intended for walled-off research computing environments where security concerns of running software are less pressing. Our users aren’t intentionally trying to misuse tools, and our tools aren’t exposed to the outside world.

We like to think of AI coding agents as tool like a “faster keyboard” rather than just “vibe-coding” features. We also acknowledge that the area between “autocomplete” and “vibe-code” is a spectrum (and some might argue a slippery slope). Drawing a line in such a vague grey area is difficult, so we tend to stick to the following overarching policy.

## Policy
**All code (AI-generated and human-written) must be thoroughly reviewed before merge by a human reviewer.** We pride ourselves on a deep understanding of what is going into Nereid and want to ensure that contributions are of a high quality.

This creates a review burden on human maintainers, thus we ask that human contributors accept the responsibility of ensuring that contributions are maintainable and reviewable. We ask that contributors effectively manage the size of PRs (avoid superfluous code additions), take care to ensure that the purpose of changes is well-described and documented, and display transparency on AI usage in accordance with the requirements below.

**Any contributions that include AI-generated code must also include a disclosure.** This can be as simple as including tagging them on the git commit/pr, but helps us keep track of where to focus our attention when things do go wrong.
