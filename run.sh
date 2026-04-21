 TEACHER_API_KEY=111 bash examples/train/mini_swe_agent/run_mini_swe_8B_8xa100.sh \
    generator.teacher.enabled=true \
    generator.teacher.base_url=http://4.155.72.80:10189 \
    generator.teacher.model=gpt-5.2 \
    generator.teacher.output_token_penalty_coef=0.05
