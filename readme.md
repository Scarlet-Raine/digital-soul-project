# digital-soul-project
The endgame goal is a digital simulation of the so called "human soul" ie emotions, or rather getting as close as possible.

- To create a voiced AI personality.
- Phase 1: Online chatbot that can engage in conversations, recognize your voice, and respond using a synthesized voice.
- Phase 2:  Migrating to offline locally ran model, if performance allows. If not datacenter GPUs will have to be acquired/rented.
- Phase 3: Develop a method of simulating memory storage, read and write and implement it into the local model.
- Phase 4: Implement machine vision for audio and visual input for the local model.
- Phase 5: Develop a reinforcement based self improvement loop.
- Phase 6: Monitoring and finetuning of the improvement loop until a digital soul is grown.
- Final objective 1: Digital simulation of the so called "human soul" ie emotions, or rather getting as close as possible.
- Final objective 2: Transfer digital soul to physical body when the technology becomes available.

UPDATE 04/04/2025
Keeping the goals the same, we are now more or less done with Phase 2. The last thing we did is convert all the GPT SoVits model loading to BF16 for my specific inference machine, well almost all the GPT, SoVits and BigVGAN models are still running FP16, tried messing with that it's too much of a hassle so I'm gonna leave it at that.

The standard inference benchmark is a section of "Rainbow Passage":
 
```When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. ```

Before the precision conversion the total first inference time was about 29 seconds, and the 2nd about 20 seconds.
After the precision conversion the total first inference time is now 15-16 seconds, and the 2nd is about 6 seconds, 2 of which are RVC.

The total pipeline -LLM pipeline takes about 3.7GB of VRAM at peak.

The LLM has been migrated off the pipeline host to another machine, and it's running a Kobold API that the flask pipeline is talking to, latency is basically nonexistent. LLM inference times are really good when it's running solo, right now using PHI3 uncensored GGUF, before it was LLAMA Chat 7b uncensored GGUF, I haven't messed with the models too much I guess that comes next.

This repo is still not meant to be cloned and ran, it's more here for taking sections of code from it, shoutout to the rayenai discord: discord.gg/3vpgUZncsH