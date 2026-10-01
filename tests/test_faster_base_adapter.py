import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Run adapter/sampling tests in the isolated Faster environment")
from service.faster_qwen3tts_base_adapter import FasterBaseAdapter, GenerationCancelled, GenerationTruncated, classify_completion

CONTROLS=dict(do_sample=True,top_k=50,top_p=1.,temperature=.9,repetition_penalty=1.05,subtalker_dosample=True,subtalker_top_k=50,subtalker_top_p=1.,subtalker_temperature=.9,non_streaming_mode=True,max_new_tokens=128)
class PolicyTests(unittest.TestCase):
    def test_eos_only_and_static_limit(self):
        self.assertEqual(classify_completion(40,128,100,2048),'eos')
        for limits in [(128,128,100,2048),(1948,2048,100,2048),(1,128,2048,2048)]:
            with self.assertRaises(GenerationTruncated):classify_completion(*limits)
    def test_empty_invalid_or_boolean_counts(self):
        for limits in [(0,128,100,2048),(1,-1,100,2048),(1,128,-1,2048)]:
            with self.assertRaises(ValueError):classify_completion(*limits)
        with self.assertRaises(TypeError):classify_completion(True,128,100,2048)
    def test_missing_cuda_no_fallback(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):
                FasterBaseAdapter.load(Path(r'C:\Users\brian\omni-audiobook\ai-audio-studio\models\qwen3_tts\Qwen3-TTS-Base'))
    def test_relative_snapshot_refused(self):
        with self.assertRaises(ValueError):FasterBaseAdapter.load(Path('Qwen/Base'))
    def test_malformed_controls_and_single_worker(self):
        adapter=FasterBaseAdapter(None)
        for controls in [dict(CONTROLS,top_p=.8),dict(CONTROLS,subtalker_temperature=.8),dict(CONTROLS,max_new_tokens=4096)]:
            with self.assertRaises(ValueError):adapter.generate_voice_clone(text='Hello',language='English',voice_clone_prompt=[None],**controls)
        adapter.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'serial'):adapter.generate_voice_clone(text='Hello',language='English',voice_clone_prompt=[None],**CONTROLS)
        finally:adapter.lock.release()
    def test_cancel_before_work_and_release(self):
        adapter=FasterBaseAdapter(None);adapter.cancel_event.set()
        with self.assertRaises(GenerationCancelled):adapter.generate_voice_clone(text='Hello',language='English',voice_clone_prompt=[None],**CONTROLS)
        self.assertFalse(adapter.lock.locked())
    def test_malformed_text_prompt(self):
        adapter=FasterBaseAdapter(None)
        for text,prompt in [('',[None]),(' ',[None]),('x'*10001,[None]),('hi',[]),('hi',[None,None])]:
            with self.assertRaises(ValueError):adapter.generate_voice_clone(text=text,language='English',voice_clone_prompt=prompt,**CONTROLS)

class SamplingTests(unittest.TestCase):
    def test_nucleus_boundary_difference_executable(self):
        from faster_qwen3_tts.sampling import sample_logits
        from transformers import TopPLogitsWarper
        logits=torch.tensor([[.6,.3,.1]]).log()
        hf=torch.softmax(TopPLogitsWarper(.8)(None,logits),-1)
        observed=[]
        def draw(p,*a,**k):observed.append(p);return torch.tensor([[0]])
        with patch('torch.multinomial',side_effect=draw):sample_logits(logits,temperature=1.,top_k=0,top_p=.8,do_sample=True)
        self.assertGreater(hf[0,1].item(),0.)
        self.assertEqual(observed[0][0,1].item(),0.)
    def test_production_top_p_one_distribution_parity(self):
        from faster_qwen3_tts.sampling import sample_logits
        from transformers import TemperatureLogitsWarper, TopKLogitsWarper
        logits=torch.linspace(-3,4,101).reshape(1,-1)
        expected=torch.softmax(TopKLogitsWarper(50)(None,TemperatureLogitsWarper(.9)(None,logits)),-1)
        observed=[]
        with patch('torch.multinomial',side_effect=lambda p,*a,**k:(observed.append(p) or torch.tensor([[0]]))):sample_logits(logits,temperature=.9,top_k=50,top_p=1.,do_sample=True)
        torch.testing.assert_close(expected,observed[0])

if __name__=='__main__':unittest.main()
