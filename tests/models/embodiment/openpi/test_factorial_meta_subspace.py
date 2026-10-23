import unittest

import torch

from rlinf.models.embodiment.openpi.factorial_meta_subspace import (
    FactorialMetaSubspaceRuntime,
    SupportConditionedResidualMask,
    SupportTaskEncoderConfig,
    TaskResidualMaskConfig,
    _private_seed,
)


def make_adapter(intervention_type):
    return SupportConditionedResidualMask(
        SupportTaskEncoderConfig(feature_dim=8, hidden_dim=6, task_code_dim=4),
        TaskResidualMaskConfig(
            intervention_type=intervention_type,
            site="pre_joint_layer_15",
            hidden_dim=8,
            task_code_dim=4,
            conditioner_hidden_dim=5,
            residual_hidden_dim=3,
        ),
    )


class FactorialMetaSubspaceTest(unittest.TestCase):
    def test_arm_parameter_topology(self):
        b = make_adapter("task_residual_mlp").intervention
        c1 = make_adapter("task_residual_mask").intervention
        c3 = make_adapter("task_residual_mask_c3").intervention
        self.assertFalse(hasattr(b, "mask"))
        self.assertFalse(hasattr(b, "delta_condition_proj"))
        self.assertTrue(hasattr(c1, "mask"))
        self.assertFalse(hasattr(c1, "delta_condition_proj"))
        self.assertTrue(hasattr(c3, "mask"))
        self.assertTrue(hasattr(c3, "delta_condition_proj"))

    def test_identity_initialization_and_token_scope(self):
        adapter = make_adapter("task_residual_mask").intervention
        hidden = torch.randn(2, 5, 8)
        code = torch.randn(2, 4)
        image_mask = torch.tensor(
            [[True, True, False, False, False], [True, False, True, False, False]]
        )
        output = adapter(
            hidden,
            task_code=code,
            image_token_mask=image_mask,
            strength=1.0,
        )
        torch.testing.assert_close(output, hidden)

        logits = torch.ones(2, 8)
        output = adapter(
            hidden,
            task_code=code,
            image_token_mask=image_mask,
            strength=1.0,
            mask_logits=logits,
        )
        torch.testing.assert_close(output[~image_mask], hidden[~image_mask])
        self.assertFalse(torch.equal(output[image_mask], hidden[image_mask]))

    def test_runtime_model_back_reference_is_not_registered(self):
        runtime = FactorialMetaSubspaceRuntime.__new__(
            FactorialMetaSubspaceRuntime
        )
        torch.nn.Module.__init__(runtime)
        runtime._model_ref = None

        policy = torch.nn.Module()
        policy.meta_subspace_runtime = runtime
        runtime._bind_model(policy)

        self.assertIs(runtime.model, policy)
        self.assertNotIn("model", runtime._modules)
        self.assertEqual(list(policy.modules()), [policy, runtime])
        policy.to("cpu")

    def test_private_seed_is_stable_and_scoped(self):
        value = [
            "libero-plus-task-residual-mask-factorial-v1",
            7,
            "camera_rlinf/episode-1.npz",
            "support-chunk",
        ]
        self.assertEqual(_private_seed(value), _private_seed(value))
        self.assertNotEqual(_private_seed(value), _private_seed([*value[:-1], "support-flow"]))


if __name__ == "__main__":
    unittest.main()
