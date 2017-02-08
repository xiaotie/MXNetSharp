using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MxNetSharp.Rcnn
{
    public class Config
    {
        public static float EPS = 0.00000000000001f;
        public static float[] PIXEL_MEANS = new float[] { 123.68f, 116.779f, 103.939f };
        public static float SCALES = 600;
        public static float MAX_SIZE = 1000;
        public static bool USE_GPU_NMS = true;
        public static int GPU_ID = 0;

        public class Train
        {
            public static bool FINETUNE = false;
            public static int BATCH_SIZE = 128;
            public static bool HAS_RPN = false;
            public static bool ASPECT_GROUPING = true;
            public static int BATCH_IMAGES = 2;
            public static float FG_FRACTION = 0.25f;
            public static float FG_THRESH = 0.5f;
            public static float BG_THRESH_HI = 0.5f;
            public static float BG_THRESH_LO = 0.1f;

            public static float BBOX_REGRESSION_THRESH = 0.5f;
            public static float[] BBOX_INSIDE_WEIGHTS = new float[] { 1.0f, 1.0f, 1.0f, 1.0f };

            public static int RPN_BATCH_SIZE = 256;
            public static float RPN_FG_FRACTION = 0.5f;
            public static float RPN_POSITIVE_OVERLAP = 0.7f;
            public static float RPN_NEGATIVE_OVERLAP = 0.3f;
            public static bool RPN_CLOBBER_POSITIVES = false;
            public static float[] RPN_BBOX_INSIDE_WEIGHTS = new float[] { 1.0f, 1.0f, 1.0f, 1.0f };
            public static float RPN_POSITIVE_WEIGHT = -1.0f;

            public static float RPN_NMS_THRESH = 0.7f;
            public static int RPN_PRE_NMS_TOP_N = 12000;
            public static int RPN_POST_NMS_TOP_N = 6000;
            public static int RPN_MIN_SIZE = 16;

            public static bool BBOX_NORMALIZATION_PRECOMPUTED = false;
            public static float[] BBOX_MEANS = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
            public static float[] BBOX_STDS = new float[] { 0.1f, 0.1f, 0.2f, 0.2f };
        }

        public class Test
        {
            public static bool HAS_RPN = false;
            public static int BATCH_IMAGES = 1;
            public static float NMS = 0.3f;
            public static float DEDUP_BOXES = 1.0f / 16.0f;
            public static float RPN_NMS_THRESH = 0.7f;
            public static int RPN_PRE_NMS_TOP_N = 6000;
            public static int RPN_POST_NMS_TOP_N = 300;
            public static int RPN_MIN_SIZE = 16;
        }
    }
}
