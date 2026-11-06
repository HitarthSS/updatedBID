# Integration and arm definitions

`AdaptiveBID().step(strong, weak, obs, prior, active)` wraps the authors' strong/weak samplers. Create the harness for the corresponding episode group, pass the original observations/prior plan and active mask, and execute one selected action. The six/fifteen candidate decision uses preceding confidence and public motion; native BID ranking and 100-step DDPM sampling are retained.

The authors' arm is the native fixed-candidate BID call. `reference/run.py` records group batching and the actual sampler binding. It depends on the pinned upstream `diffusion_policy` package, two original checkpoint files, initial-state manifest and source-freeze receipts. Those bulky/private runtime assets are deliberately not bundled; the driver is not advertised as a fresh-machine launcher. Primary mean maximum overlap and thresholded complete success are different metrics.
