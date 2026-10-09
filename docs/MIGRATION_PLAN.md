# CL11 移行計画

対象: `adapters/ggml/ggml-cl11.cpp` 4033 行、旧 `adapters/ggml/kernels/*.cl`（現在は `src/backends/opencl11/kernels/*.cl`）、`adapters/ggml/CMakeLists.txt`。根拠は現行ファイルの行番号。提案は「提案」と明記する。

## 1. `ggml-cl11.cpp` インベントリ

### 型・テーブル・グローバル

| 行 | 項目 | 目的 | 依存 | 移設先 |
| --- | --- | --- | --- | --- |
| 45 | `CL11_NUM_SOURCES` | CMake 生成 kernel 配列の個数。 | `ggml_cl11_sources` | backends/opencl11 / kernels |
| 49 | `CL11_BASE` | ggml buffer の仮想 base pointer。 | ggml buffer API | adapters/ggml、core buffer には抽象 offset として分離 |
| 52 | `CL11_MAX_BUFFER_SIZE` | kernel の 32-bit byte offset 上限。 | OpenCL buffer / kernels | core/runtime + backends/opencl11 |
| 55 | `CL11_MIN_ALIGNMENT` | vector load と `CL11_BASE` 用 alignment。 | kernels / ggml allocator | backends/opencl11 buffer、adapter shim |
| 61-85 | `cl11_device_info` | context 作成前の device 能力。 | OpenCL device/platform info | core/runtime device + backends/opencl11 device |
| 88-94 | `cl11_rkind` | R32 repack 対象量子型の内部分類。 | `GGML_TYPE_Q*` 変換関数 | core/quant。ただし ggml enum 変換は adapter |
| 97-108 | `cl11_mmx` | parameterized mat-mat variant。 | tuning/env/kernels | backends/opencl11 tuning/kernels |
| 111-129 | `cl11_config` | work-group と kernel compile 設定。 | device limits/env/kernels | backends/opencl11 tuning/executor |
| 137-148 | `cl11_tuning` | GPU 世代別既定 tuning。 | `cl11_device_info` | backends/opencl11 tuning |
| 151-161 | `cl11_kernel_class` | kernel work-group 制限の分類。 | kernel table | backends/opencl11 kernels |
| 164-172 | `cl11_qtype` | kernel variant 用 dtype index。 | ggml dtype 変換 | core/tensor dtype + adapter 変換 |
| 174-211 | `cl11_kernel_id` | host 側 kernel handle index。 | kernel files/table | backends/opencl11 kernels |
| 226-229 | `cl11_kernel_desc` | kernel 名と class。 | kernel table | backends/opencl11 kernels |
| 232-345 | `cl11_kernel_table[]` | `K_*` から OpenCL kernel 名へ対応。 | `cl11_kernel_id` / `.cl` 名 | backends/opencl11 kernels |
| 349-355 | `cl11_td` | kernel ABI の tensor descriptor。 | kernels/common.cl `tdesc`、ggml fields | core/tensor + backend ABI。ggml 読み取りは adapter |
| 372-379 | `cl11_bcast` | kernel ABI の broadcast descriptor。 | `cl11_fastdiv_init` / kernels | core/tensor view/stride |
| 395-497 | `cl11_device_context` | device ごとの lazy context/program/kernel/cache/profiling/ggml device。 | OpenCL API、ggml device/buft、env、CL11 helpers | split needed: OpenCL state は backends/opencl11、ggml `buft/dev` は adapters/ggml、profile/cache は executor |
| 499 | `g_cl11_devices` | process global device registry。 | `cl11_probe_devices` / ggml registry | backends/opencl11 device registry + adapter registry shim |
| 595-600 | `cl11_amd_gen` | AMD generation 分類。 | device info strings | backends/opencl11 tuning |
| 700-703 | `cl11_mmx_preset` | MMX preset entry。 | parser/table | backends/opencl11 tuning |
| 706-719 | `cl11_mmx_presets[]` | Kepler mat-mat preset 一覧。 | env parser / kernels | backends/opencl11 tuning |
| 1609-1635 | `cl11_args` | `clSetKernelArg` 連鎖 helper。 | OpenCL kernel、`cl11_err_str` | backends/opencl11 executor |
| 1719-1723 | `cl11_buffer_context` | ggml buffer context から `cl_mem` を持つ。 | ggml buffer、OpenCL mem | split needed: CL buffer は backend、ggml context は adapter |
| 1734-1737 | `cl11_tref` | `cl_mem + cl11_td` の launch 参照。 | `cl11_td` / `cl_mem` | backends/opencl11 executor/tensor handle |
| 1801-1805 | `cl11_rp_area` | R32 area 定義。 | repack algorithm | core/quant |
| 1807-1814 | `cl11_rp_geom` | R32 geometry。 | ggml dtype/block size 現状依存 | core/quant。dtype/block size は core enum に置換 |
| 2129-2141 | `ggml_backend_cl11_buffer_i` | ggml buffer vtable。 | ggml backend API | adapters/ggml |
| 2257-2266 | `ggml_backend_cl11_buft_i` | ggml buffer type vtable。 | ggml backend API | adapters/ggml |
| 3087-3092 | `cl11_padp` | `PAD` kernel ABI params。 | ggml op params / moe_ops.cl | core/ops + backend ABI。ggml param decode は adapter |
| 3515-3518 | `cl11_backend_context` | ggml backend instance state。 | ggml backend API、device context | adapters/ggml |
| 3798-3815 | `ggml_backend_cl11_i` | ggml backend vtable。 | ggml backend API | adapters/ggml |
| 3913-3929 | `ggml_backend_cl11_device_i` | ggml device vtable。 | ggml backend API | adapters/ggml |
| 3951-3956 | `ggml_backend_cl11_reg_i` | ggml registry vtable。 | ggml backend API | adapters/ggml |

### 関数

| 行 | 項目 | 目的 | 依存 | 移設先 |
| --- | --- | --- | --- | --- |
| 213-216 | `cl11_is_k_kernel` | K-quant 別 program 対象判定。 | `K_*` | backends/opencl11 kernels |
| 218-220 | `cl11_is_mmx_kernel` | MMX kernel 範囲判定。 | `K_*` | backends/opencl11 kernels |
| 222-224 | `cl11_is_mm_kernel` | mat-mat kernel 範囲判定。 | `K_*` | backends/opencl11 kernels |
| 359-369 | `cl11_fastdiv_init` | kernel 用 fast division magic 生成。 | なし | core/tensor |
| 382-393 | `cl11_make_bcast` | broadcast descriptor 生成。 | `cl11_fastdiv_init` | core/tensor |
| 501-504 | `cl11_debug` | `GGML_CL11_DEBUG` cache。 | env | backends/opencl11 tuning/log |
| 506-532 | `cl11_err_str` | OpenCL error 表示。 | OpenCL error enum | backends/opencl11 runtime |
| 543-549 | `cl11_pow2_floor` | tuning 用 pow2 floor。 | なし | core/runtime utility または backend tuning |
| 555-568 | `cl11_device_str` | OpenCL device string 取得。 | `clGetDeviceInfo` | backends/opencl11 device |
| 571-577 | `cl11_device_val<T>` | OpenCL device scalar 取得。 | `clGetDeviceInfo` | backends/opencl11 device |
| 579-584 | `cl11_lower` | env/device name 比較用 lowercase。 | std | core/runtime utility または backend tuning |
| 602-694 | `cl11_amd_generation` | AMD 世代推定。 | device strings | backends/opencl11 tuning |
| 723-793 | `cl11_mmx_parse` | `GGML_CL11_MM_VARIANT/MMX` parse。 | presets、env 文字列 | backends/opencl11 tuning |
| 796-819 | `cl11_mmx_valid` | MMX variant の WG/local mem 検証。 | device limits | backends/opencl11 tuning |
| 821-911 | `cl11_tuning_for` | Fermi/Kepler/AMD 既定 tuning。 | `cl11_device_info`、`GGML_CL11_FORCE_GEN` | backends/opencl11 tuning |
| 914-939 | `cl11_device_selected` | `GGML_CL11_DEVICES` filter。 | env、lower | backends/opencl11 device |
| 941-1023 | `cl11_probe_devices` | OpenCL platform/device 列挙と `g_cl11_devices` 追加。 | OpenCL API、env、tuning、global | split needed: backend device list + adapter registry init |
| 1032-1076 | `cl11_dn_probe` | Fermi denormal product 可否 probe。 | OpenCL program/kernel/buffer/queue | backends/opencl11 tuning/probe |
| 1081-1185 | `cl11_build_program` | OpenCL source build、opts、kernel dir/dump。 | OpenCL API、generated kernels、env | backends/opencl11 context/kernels |
| 1187-1207 | `cl11_release_kernels` | kernel/program release。 | OpenCL API、context fields | backends/opencl11 context |
| 1210-1278 | `cl11_ensure_context` | lazy context/queue 作成、env runtime 設定。 | OpenCL API、env、tuning | split needed: context は backend、ggml 初期化呼び出しは adapter |
| 1283-1549 | `cl11_ensure_program` | program build、WG 縮小、kernel handle 作成。 | OpenCL API、env、tuning、kernel table | backends/opencl11 executor/tuning |
| 1552-1603 | `cl11_ensure_kprogram` | K-quant 別 program 遅延 build。 | OpenCL API、kernel table | backends/opencl11 kernels |
| 1638-1649 | `cl11_prof_collect` | event profiling 集計。 | OpenCL event、profile state | backends/opencl11 executor |
| 1651-1660 | `cl11_prof_key` | profile key 検索/追加。 | `std::vector<std::string>` | backends/opencl11 profiling。hot path では plan cache 化 |
| 1662-1683 | `cl11_prof_print` | profile 表示。 | profile state | backends/opencl11 profiling |
| 1687-1709 | `cl11_launch` | 2-D NDRange enqueue。 | OpenCL queue/kernel、profile | backends/opencl11 executor |
| 1711-1713 | `cl11_div_up` | ceil division。 | なし | core utility |
| 1725-1727 | `cl11_tensor_buffer` | ggml view 元 buffer 解決。 | `ggml_tensor::view_src/buffer` | adapters/ggml |
| 1730-1732 | `cl11_tensor_offset` | ggml dummy base から byte offset 算出。 | `ggml_tensor::data`、`CL11_BASE` | adapters/ggml、core は explicit offset |
| 1739-1752 | `cl11_ref` | ggml tensor から launch 参照作成。 | ggml buffer/ne/nb/data、`cl11_fastdiv_init` | split needed: adapter が core TensorDesc を作り backend handle と結合 |
| 1773-1775 | `cl11_rp_type` | ggml quant type が R32 対象か判定。 | `GGML_TYPE_*` | adapter dtype 変換 + core/quant |
| 1777-1784 | `cl11_rkind` | ggml dtype から R32 kind。 | `GGML_TYPE_*` | adapter dtype 変換 |
| 1786-1792 | `cl11_repacked` | tensor が R32 layout か判定。 | ggml tensor fields、`ggml_is_contiguous` | split needed: core layout predicate + adapter mapping |
| 1795-1799 | `cl11_rp_check_view` | R32 view 禁止の guard。 | ggml view/name | adapters/ggml guard、core は view constraint |
| 1816-1829 | `cl11_rp_geometry` | ggml tensor から R32 geometry。 | ggml type/block/shape | core/quant、adapter は dtype/shape を渡す |
| 1832-1838 | `cl11_rp_area_before` | area offset 計算。 | R32 geom | core/quant |
| 1841-1867 | `cl11_rp_pack` | natural -> R32 pack。 | R32 geom | core/quant |
| 1870-1896 | `cl11_rp_unpack` | R32 -> natural unpack。 | R32 geom | core/quant |
| 1900-1964 | `cl11_rp_transfer` | R32 部分転送と read-modify-write。 | OpenCL read/write、ggml tensor、pack/unpack | split needed: quant transfer は core/quant + backend buffer、ggml tensor は adapter |
| 1967-2005 | `cl11_fill` | OpenCL 1.1 用 memset kernel。 | `K_FILL_*`、OpenCL launch | backends/opencl11 buffer/executor |
| 2007-2025 | `ggml_backend_cl11_buffer_free_buffer` | ggml buffer free、sub-buffer 新しい順 release。 | ggml buffer、OpenCL release、const cache | split needed: backend release order + adapter vtable |
| 2027-2030 | `ggml_backend_cl11_buffer_get_base` | ggml allocator 用 dummy base。 | `CL11_BASE` | adapters/ggml |
| 2032-2038 | `ggml_backend_cl11_buffer_memset_tensor` | ggml tensor range clear。 | ggml buffer/tensor、`cl11_fill` | adapters/ggml shim -> backend buffer fill |
| 2040-2058 | `ggml_backend_cl11_buffer_set_tensor` | host -> device。R32 なら repack。 | ggml tensor、OpenCL write、quant | split needed |
| 2060-2077 | `ggml_backend_cl11_buffer_get_tensor` | device -> host。R32 なら unpack。 | ggml tensor、OpenCL read、quant | split needed |
| 2079-2119 | `ggml_backend_cl11_buffer_cpy_tensor` | host/device/device copy。 | ggml buffer internals、OpenCL copy | split needed |
| 2121-2127 | `ggml_backend_cl11_buffer_clear` | whole buffer clear。 | ggml buffer、`cl11_fill` | adapters/ggml shim |
| 2148-2150 | `cl11_max_buffer_size` | 公称 buffer 上限。 | device info、CL11 max | core/runtime buffer |
| 2153-2159 | `cl11_strict_alloc` | `GGML_CL11_STRICT_ALLOC` cache。 | env | backends/opencl11 allocator config |
| 2164-2169 | `cl11_max_alloc_size` | 実 alloc 上限。 | strict/device/global mem | core/runtime allocator + backend |
| 2171-2174 | `ggml_backend_cl11_buft_get_name` | ggml buft name。 | ggml buft context | adapters/ggml |
| 2177-2187 | `cl11_probe_buffer` | lazy allocation を端 write で検出。 | OpenCL write | backends/opencl11 allocator |
| 2189-2236 | `ggml_backend_cl11_buft_alloc_buffer` | ggml buffer alloc、probe。 | ggml buffer init、OpenCL create/probe | split needed |
| 2238-2245 | `ggml_backend_cl11_buft_get_alignment` | ggml alignment 報告。 | device align、CL11 constants | adapters/ggml shim + backend buffer props |
| 2247-2250 | `ggml_backend_cl11_buft_get_max_size` | ggml alloc split 用 max size。 | device context | adapters/ggml shim |
| 2252-2255 | `ggml_backend_cl11_buft_is_host` | device buffer 判定。 | ggml API | adapters/ggml |
| 2272-2282 | `cl11_qtype_index` | ggml dtype -> kernel dtype。 | `GGML_TYPE_*` | adapter dtype conversion |
| 2284-2286 | `cl11_is_float` | ggml F32/F16 判定。 | `GGML_TYPE_*` | adapter dtype conversion |
| 2289-2300 | `cl11_fits_int` | 32-bit descriptor 制約判定。 | ggml tensor/ne/nb/nbytes | core/tensor validation + adapter mapping |
| 2303-2309 | `cl11_sort_npad` | sort row padding。 | なし | core/ops |
| 2311-2313 | `cl11_sort_fits` | local memory に sort が収まるか。 | device local mem | core/ops + backend capabilities |
| 2316-2322 | `cl11_op_disabled` | `GGML_CL11_NO_OPS` fallback forcing。 | env、`ggml_op_desc` | adapters/ggml debug |
| 2324-2509 | `cl11_supports_op_impl` | ggml op/type/shape support 判定。 | ggml op enum/type/shape/op_params、device limit | split needed: adapter translates; core validates op contracts; backend reports capabilities |
| 2515-2532 | `cl11_op_binary` | ADD/SUB/MUL/DIV launch。 | ggml op/src, `cl11_ref`, kernels | core/ops + backend executor。ggml decode は adapter |
| 2535-2574 | `cl11_op_unary` | UNARY/SCALE/CLAMP/SQR/SQRT launch。 | ggml op params/enums | split needed |
| 2576-2591 | `cl11_op_glu` | GLU launch。 | ggml GLU params | split needed |
| 2593-2603 | `cl11_op_rms_norm` | RMS_NORM launch。 | ggml op params | core/ops + backend |
| 2607-2624 | `cl11_norm_mul_weights` | RMS_NORM+MUL fusion 判定。 | `ggml_cgraph` adjacency/use counts | core/graph plan または adapters/ggml plan translator |
| 2626-2636 | `cl11_op_rms_norm_mul` | fused RMS_NORM+MUL launch。 | ggml tensors/op params | core/ops fused op + backend |
| 2639-2657 | `cl11_can_fuse_add_norm_mul` | ADD+RMS_NORM+MUL fusion 判定。 | `ggml_cgraph` adjacency/use counts | core/graph plan / adapter |
| 2659-2667 | `cl11_op_add_rms_norm_mul` | fused ADD+RMS_NORM+MUL launch。 | ggml tensors/op params | core/ops fused op + backend |
| 2669-2693 | `cl11_op_soft_max` | SOFT_MAX launch。 | ggml mask/op params | core/ops + backend |
| 2695-2730 | `cl11_op_rope` | ROPE launch。 | ggml ROPE params、`ggml_rope_yarn_corr_dims` | split needed: formula/core op; backend launch |
| 2732-2740 | `cl11_op_get_rows` | GET_ROWS launch。 | ggml src/type | core/ops + backend |
| 2742-2754 | `cl11_op_set_rows` | SET_ROWS launch。 | ggml I32/I64 type | core/ops + backend |
| 2757-2770 | `cl11_op_set_rows2` | paired SET_ROWS fused launch。 | ggml tensors | core/ops fused op + backend |
| 2773-2792 | `cl11_can_pair_set_rows` | SET_ROWS fusion safety。 | ggml views/src | core/graph plan / adapter |
| 2794-2817 | `cl11_op_cpy` | copy/transpose-copy launch。 | ggml contiguity/shape/type | split needed |
| 2820-2829 | `cl11_xprep_scratch` | Fermi DN src1 prep scratch alloc。 | OpenCL buffer、device context | backends/opencl11 executor cache |
| 2832-2851 | `cl11_xprep_note_write` | scratch / MID ids cache invalidation。 | ggml node data/nbytes、CL refs | split needed: core execution hazards + adapter tensor refs |
| 2855-2872 | `cl11_const_sub` | constant sub-buffer cache。 | OpenCL sub-buffer、device align | backends/opencl11 buffer/cache |
| 2874-2880 | `cl11_log2_exact` | exact log2 for launch params。 | なし | core utility |
| 2885-2897 | `cl11_attn_ok` | attention specialized kernel 条件。 | CL11 desc/config | core/ops capability + backend |
| 2901-3051 | `cl11_mul_mat_core` | mat-vec/mat-mat/attention/R32 path selectionと launch。 | ggml dtype、CL11 desc、const/xprep/kprogram | split needed: core op selection contract + backend executor; adapter dtype 変換 |
| 3053-3055 | `cl11_op_mul_mat` | ggml MUL_MAT wrapper。 | ggml tensor src | adapters/ggml |
| 3059-3065 | `cl11_op_add_id` | ADD_ID launch。 | ggml tensors | core/ops + backend |
| 3067-3073 | `cl11_op_ssm_conv` | SSM_CONV launch。 | ggml tensors | core/ops + backend |
| 3075-3084 | `cl11_op_concat` | CONCAT launch。 | ggml params/type size | split needed |
| 3095-3111 | `cl11_op_pad` | PAD launch。 | ggml pad params | split needed |
| 3113-3120 | `cl11_op_sum_rows` | SUM_ROWS launch。 | ggml shape | core/ops + backend |
| 3122-3131 | `cl11_op_argsort` | ARGSORT/TOP_K launch。 | ggml sort params | core/ops + backend |
| 3136-3151 | `cl11_td_matrix` | sub-matrix descriptor 作成。 | `cl11_td` | core/tensor |
| 3153-3177 | `cl11_mid_ensure` | MUL_MAT_ID scratch buffer grow/probe。 | OpenCL buffer/probe | backends/opencl11 buffer/executor |
| 3182-3282 | `cl11_op_mul_mat_id_sync` | prompt path: ids readback、gather、expert matmul、scatter。 | ggml tensors/strides、OpenCL read/write、scratch | split needed: core MoE op plan + backend scratch; ggml ids layoutは adapter |
| 3286-3345 | `cl11_op_mul_mat_id_gen` | generation path: GPU ids 読み mat-vec。 | ggml tensors、R32 path | core/ops + backend |
| 3347-3358 | `cl11_op_mul_mat_id` | MID path selection。 | `GGML_CL11_MID_MAX_TOKENS` state、ggml shape | core/ops policy + backend |
| 3364-3375 | `cl11_mvm_candidate` | multi mat-vec 候補判定。 | ggml op/type/view/shape | core/graph plan / adapter |
| 3378-3381 | `cl11_mvm_same_group` | multi mat-vec group 判定。 | ggml src identity/type/shape | core/graph plan / adapter |
| 3384-3419 | `cl11_op_mul_mat_multi` | multi mat-vec fused launch。 | ggml group、R32、kprogram | core/ops fused op + backend |
| 3422-3439 | `cl11_op_mul_mat_t` | CONT skipped V-cache mat-vec launch。 | ggml transposed view、CL11 desc | core/ops fused op + backend |
| 3442-3474 | `cl11_can_skip_cont` | CONT+MUL_MAT fusion 判定。 | `ggml_cgraph` adjacency/use counts | core/graph plan / adapter |
| 3476-3509 | `cl11_compute_node` | ggml op enum dispatch。 | ggml op enum、op functions | adapters/ggml dispatch shim |
| 3520-3523 | `ggml_backend_cl11_get_name` | backend name。 | ggml backend context | adapters/ggml |
| 3525-3537 | `ggml_backend_cl11_free` | backend instance free、profile finish/print。 | ggml backend、OpenCL finish/profile | adapters/ggml + backend profiling |
| 3539-3549 | `ggml_backend_cl11_synchronize` | queue `clFinish`。 | OpenCL queue | adapters/ggml shim -> backend stream |
| 3551-3751 | `ggml_backend_cl11_graph_compute` | graph walk、fusion、dispatch、flush/profile。 | `ggml_cgraph`、ggml flags、env、all ops | split needed: adapter graph translator + core graph/execution plan + backend executor |
| 3757-3796 | `ggml_backend_cl11_graph_optimize` | multi-MV 用 node reorder。 | mutable `ggml_cgraph->nodes` | adapters/ggml、将来 core graph planner に同等機能 |
| 3817-3820 | `ggml_backend_cl11_guid` | ggml backend GUID。 | ggml GUID API | adapters/ggml |
| 3826-3828 | `cl11_dev_ctx` | ggml dev context cast。 | ggml device | adapters/ggml |
| 3830-3832 | `ggml_backend_cl11_device_get_name` | ggml device name。 | device context | adapters/ggml |
| 3834-3836 | `ggml_backend_cl11_device_get_description` | OpenCL device description。 | device info | adapters/ggml shim |
| 3838-3844 | `ggml_backend_cl11_device_get_memory` | free/total memory 報告。 | device info | backends/opencl11 device + adapter |
| 3846-3849 | `ggml_backend_cl11_device_get_type` | GPU type 報告。 | ggml enum | adapters/ggml |
| 3851-3864 | `ggml_backend_cl11_device_get_props` | ggml props 埋め。 | ggml API | adapters/ggml |
| 3866-3884 | `ggml_backend_cl11_device_init_backend` | backend instance 作成。 | ggml API、`cl11_ensure_context` | adapters/ggml |
| 3886-3888 | `ggml_backend_cl11_device_get_buffer_type` | buft 取得。 | ggml API | adapters/ggml |
| 3890-3906 | `ggml_backend_cl11_device_supports_op` | support 判定と unsupported log。 | ggml op、env、support impl | adapters/ggml |
| 3908-3911 | `ggml_backend_cl11_device_supports_buft` | same-device buft 判定。 | ggml buft internals | adapters/ggml |
| 3935-3938 | `ggml_backend_cl11_reg_get_name` | registry name。 | `GGML_CL11_NAME` | adapters/ggml |
| 3940-3943 | `ggml_backend_cl11_reg_get_device_count` | device count。 | `g_cl11_devices` | adapters/ggml registry |
| 3945-3949 | `ggml_backend_cl11_reg_get_device` | device pointer 取得。 | `g_cl11_devices` | adapters/ggml registry |
| 3958-3991 | `ggml_backend_cl11_reg` | lazy registry init、device probe、buft/dev 作成。 | ggml API、global、`cl11_probe_devices` | split needed: backend registry + ggml adapter registry |
| 3993 | `GGML_BACKEND_DL_IMPL` | ggml dynamic backend export。 | ggml macro | adapters/ggml |
| 3999-4001 | `ggml_backend_cl11_get_device_count` | public C API: device count。 | ggml registry | adapters/ggml public compatibility |
| 4003-4012 | `ggml_backend_cl11_get_device_description` | public C API: description。 | ggml registry/device API | adapters/ggml public compatibility |
| 4014-4021 | `ggml_backend_cl11_init` | public C API: backend init。 | ggml registry/device API | adapters/ggml public compatibility |
| 4023-4025 | `ggml_backend_is_cl11` | public C API: backend identity。 | ggml GUID | adapters/ggml public compatibility |
| 4027-4033 | `ggml_backend_cl11_buffer_type` | public C API: buft 取得。 | ggml registry/device API | adapters/ggml public compatibility |

## 2. core 移設前に外す ggml-specific leakage

| 種別 | 場所 | 内容 | 除去方針 |
| --- | --- | --- | --- |
| tensor field 直接参照 | 1725-1752, 1786-1792, 1816-1821, 2289-2300, 2324-2509, 2515-3509 | `ggml_tensor::ne/nb/data/buffer/view_src/src/type/op/op_params/name/flags` を core 候補が読む。 | adapter で `ScrapTensorDesc`、`ScrapBufferRef`、`ScrapOpParams` に翻訳し、core/backend は ggml pointer を受け取らない。 |
| ggml dtype enum | 1773-1784, 2272-2286, 2324-2509, op launch 群 | `GGML_TYPE_F32/F16/Q4_0/Q8_0/Q4_K/Q6_K/I32/I64/F64` に直接依存。 | `scrapgpu_dtype` に変換する adapter 関数を置き、R32 kind は core/quant enum にする。 |
| ggml op enum | 2316-2322, 2324-2509, 2519-2525, 3476-3509, 3551-3751 | `GGML_OP_*` と `ggml_op_desc/name` に依存。 | adapter が `ScrapOp` に変換。unsupported logging は adapter 側。 |
| op params layout | 2408-2428, 2540-2562, 2580-2583, 2596-2597, 2660-2661, 2673-2674, 2700-2716, 3079, 3098-3104, 3125 | ggml の `op_params` offsets と helper を読む。 | op ごとの param struct を core に定義し、adapter が ggml layout から decode。 |
| ggml helper | 1787, 1819-1820, 2082, 2299, 2369, 2376, 2388, 2395, 2455, 2473-2475, 2527, 2586, 2602, 2614, 2618-2619, 2645-2656, 2715, 2748, 2761-2762, 2800, 3061, 3078, 3107, 3565, 3582, 3651, 3703-3720, 3897 | `ggml_*` utility に広く依存。 | core utility/shape API に置換。`ggml_rope_yarn_corr_dims` は core に式を移すか adapter で算出。 |
| graph adjacency / fusion | 2607-2657, 3442-3474, 3551-3751, 3757-3796 | RMS_NORM+MUL、ADD+RMS_NORM+MUL、multi mat-vec、paired SET_ROWS、CONT skip、CONT+MUL_MAT V cache が `ggml_cgraph->nodes` と use count に依存。 | adapter が ggml graph から core `ExecutionPlan` を作る。fusion は `PlanNode` と fused op として表現。 |
| tensor->extra / buffer internals | 1725-1742, 2007-2127, 2171-2266, 3908-3911 | `buffer->context`、`buffer->iface.free_buffer`、`buft->context`、`dev->context` に依存。`tensor->extra` は本ファイルでは直接使用なし。 | backend buffer handle と adapter-owned ggml buffer context を分離。same-device 判定は opaque backend id で行う。 |
| mutable graph rewrite | 3757-3796 | `cgraph->nodes` を直接並べ替える。 | ggml adapter 内に残すか、core plan を別配列で作り ggml graph 自体は変更しない提案。 |

## 3. Global state と lifetime

| 項目 | 場所 | 現状 | 移行時の注意 |
| --- | --- | --- | --- |
| device registry | 499, 941-1023, 3958-3991 | `g_cl11_devices` に `cl11_device_context` を保存し、registry 初期化時に probe。 | core/backend の device registry と ggml registry vtable を分離。 |
| lazy context | 403-414, 1210-1278, 3866-3875 | first buffer/backend init で `clCreateContext` と in-order queue 作成。 | list devices で build しない挙動を維持。 |
| lazy program | 405-410, 1283-1549, 1552-1603 | first graph/fill で build、K-quant program は必要時。 | build result と kernel handles を backend device に cache。 |
| program caches | 406-415, 1187-1207 | `program`, `program_k`, `program_mm`, `kernels[K_COUNT]`。 | split library でも `libggml-cl11.so` 内で symbol 可視性と lifetime を明確化。 |
| constant sub-buffer cache | 439-446, 2855-2872, 2007-2019 | `(parent,offset,size)->sub` cache。free 時は NVIDIA 390 crash 回避のため新しい順に release。 | release order は backend buffer lifetime 契約に残す。 |
| `k_xprep` scratch | 448-457, 2820-2851, 2933-2958 | Fermi DN 用 64KiB scratch、書き込み overlap で invalid。 | executor cache と hazard tracking に移す。 |
| MUL_MAT_ID scratch | 459-474, 3153-3282 | gather/out/pairs buffer、ids readback cache。 | MoE executor scratch と graph-scope cache に分離。 |
| profiling | 477-493, 1638-1683, 3525-3534, 3702-3721, 3746-3748 | event を pending に積み、graph/profile free 時に collect/print。 | profiling off では event/文字列処理なしを維持。 |
| synchronization | 6-7, 110-112 in `docs/CL11.md`, 3539-3549, 3744-3749 | graph 後 `clFlush`、`synchronize()`/get/set で wait。profile 時は graph ごと finish。 | node 単位 `clFinish` を追加しない。 |

### `GGML_CL11_*` 環境変数

| 変数 | 読み取り行 | 用途 |
| --- | --- | --- |
| `GGML_CL11_DEBUG` | 501-504 | debug/build log と WG limit 表示。 |
| `GGML_CL11_FORCE_GEN` | 824-829 | Fermi/Kepler tuning 強制。 |
| `GGML_CL11_DEVICES` | 913-939 | device name substring filter。 |
| `GGML_CL11_FAKE_MAX_ALLOC` | 990-993 | max alloc debug override。 |
| `GGML_CL11_BUILD_OPTS` | 1092 | OpenCL build options 追加。 |
| `GGML_CL11_KERNEL_DIR` | 1126-1144 | embedded ではなく外部 `.cl` を読む。 |
| `GGML_CL11_DUMP_BINARY` | 1169-1179 | program binary dump。 |
| `GGML_CL11_PROFILE` | 1226 | profiling queue 有効化。 |
| `GGML_CL11_LAUNCH_LOG` | 1227 | graph ごとの launch 数出力。 |
| `GGML_CL11_MV_MAX_COLS` | 1237-1239 | mat-vec を使う src1 列数上限。 |
| `GGML_CL11_MID_MAX_TOKENS` | 1240-1242 | `MUL_MAT_ID` generation path 閾値。 |
| `GGML_CL11_MV_TPR` | 1243-1245 | generic mat-vec threads per row。 |
| `GGML_CL11_MV_STAGED` | 1246-1248 | staged mat-vec 有効/無効。 |
| `GGML_CL11_NO_CPY_T` | 1249 | transposing copy kernel 無効。 |
| `GGML_CL11_NO_MVT` | 1250 | CONT+MUL_MAT fusion 無効。 |
| `GGML_CL11_NO_MVQ` | 1251 | K-cache special kernel 無効。 |
| `GGML_CL11_NO_FUSE` | 1252, 1431 | consecutive/fusion 全般無効。 |
| `GGML_CL11_NO_ASM` | 1256 | NVIDIA inline PTX 無効。 |
| `GGML_CL11_NO_LDG` | 1257 | NVIDIA `ld.global.nc` 無効。 |
| `GGML_CL11_MVR_CONST_X` | 1266-1268, 1406 | constant-memory src1 path 強制。 |
| `GGML_CL11_MVS_FAMILY` | 1269-1271 | `k_mvx`/`k_mvw` family 強制。 |
| `GGML_CL11_MVS_MIN_K` | 1272-1274 | staged mat-vec 最小 K。 |
| `GGML_CL11_MAX_WG` | 1295-1298 | work-group cap。 |
| `GGML_CL11_MAD` | 1302-1304 | `mad()` 使用。 |
| `GGML_CL11_MVS_NRW` | 1305-1307 | staged mat-vec rows per warp。 |
| `GGML_CL11_WG_MV` | 1308-1310 | generic/staged mat-vec WG。 |
| `GGML_CL11_WG_MVR` | 1315-1321 | R32 mat-vec WG 全型。 |
| `GGML_CL11_WG_MVR_<TYPE>` / `GGML_CL11_WG_MVR_Q4_0/Q8_0/Q4_K/Q6_K` | 1315-1321 | R32 mat-vec WG 型別。 |
| `GGML_CL11_MVR_LDG` | 1322-1328 | R32 mat-vec LDG 全型。 |
| `GGML_CL11_MVR_LDG_<TYPE>` / `GGML_CL11_MVR_LDG_Q4_0/Q8_0/Q4_K/Q6_K` | 1322-1328 | R32 mat-vec LDG 型別。 |
| `GGML_CL11_MVR_DN` | 1329-1335, 1405 | Fermi denormal products 全型。 |
| `GGML_CL11_MVR_DN_<TYPE>` / `GGML_CL11_MVR_DN_Q4_0`, `GGML_CL11_MVR_DN_Q4_K`, `GGML_CL11_MVR_DN_Q6_K` | 1329-1335, 1405 | DN 型別。Q8_0 は kernel なしで 1398 に無効化。 |
| `GGML_CL11_MVR_NRR` | 1336-1342 | R32 rows per lane 全型。 |
| `GGML_CL11_MVR_NRR_<TYPE>` / `GGML_CL11_MVR_NRR_Q4_0/Q8_0/Q4_K/Q6_K` | 1336-1342 | R32 rows per lane 型別。 |
| `GGML_CL11_WG_ATT` | 1345-1348 | attention WG。 |
| `GGML_CL11_ATT_P` | 1349-1351 | K-cache positions per group。 |
| `GGML_CL11_MVT_GROUPS` | 1352-1355 | V-cache many-groups kernel。 |
| `GGML_CL11_NV_MAXRREG` | 1357-1359 | NVIDIA K-quant program register cap。 |
| `GGML_CL11_MM_BN` | 1360-1362 | mat-mat tile N factor。 |
| `GGML_CL11_MM_VARIANT` | 1368-1375 | MMX preset/spec。 |
| `GGML_CL11_MMX` | 1368-1379 | MMX spec 追加 override。 |
| `GGML_CL11_DN_NOPROBE` | 1413 | DN probe skip。 |
| `GGML_CL11_SPLIT_PROGRAMS` | 1420-1422 | decode/mm/K programs 分割。 |
| `GGML_CL11_MULTI_MV` | 1427-1430 | multi mat-vec fusion。 |
| `GGML_CL11_FUSE_SET_ROWS` | 1431-1434 | paired SET_ROWS fusion。 |
| `GGML_CL11_STRICT_ALLOC` | 2152-2159, 2197 | max alloc 以上を拒否。 |
| `GGML_CL11_NO_OPS` | 2315-2322 | 指定 ggml op を unsupported にする。 |
| `GGML_CL11_TRACE_GRAPH` | 3560-3569 | 2 回目 graph の先頭 80 node 出力。 |
| `GGML_CL11_PROFILE_SHAPES` | 3710-3716 | profile key に shape/stride 追加。 |
| `GGML_CL11_LOG_UNSUPPORTED` | 3892-3903 | unsupported op を一度だけ出力。 |

`GGML_CL11_NAME` は 3937 で registry 名に使う macro で、環境変数ではない。

## 4. 回帰禁止の hot path

| hot path | 場所 | 回帰要因 | 回避策 |
| --- | --- | --- | --- |
| per-node dispatch | 3476-3509, 3551-3751 | virtual call、文字列生成、map lookup、allocation を node ごとに追加すると decode の数百 launch に効く。 | adapter で graph から `ExecutionPlan` を作り、node ごとに resolved function/kernel id/arg layout を cache。 |
| argument setting | 1609-1635, 各 `cl11_op_*` | type-erased variant、heap allocation、`std::function` は `clSetKernelArg` 前に overhead。 | POD arg builder を stack で使い、kernel ABI struct は固定。 |
| launch | 1687-1709 | launch wrapper の virtual indirection、event 作成、文字列処理。 | profile off では event null、resolved queue/kernel/local/groups を直接渡す。 |
| mat-vec selection | 2901-3051, 3286-3345, 3384-3419 | selection を launch ごとに高価な map/string/device query にすると tg が落ちる。 | dtype/layout/shape/device から kernel variant を plan 時に決定。dynamic な alignment だけ fast branch。 |
| R32 addressing | 1754-1964, 2914-2968 | generic layout 抽象で per-weight address 計算や全量 dequant を入れると失敗。 | R32 は `PhysicalLayout` として保持し、既存 `.cl` の addressing と host pack/unpack を維持。 |
| fusion | 2607-2667, 3364-3474, 3551-3796 | planner を失うと launch 数増。`docs/CL11.md` では Kepler fusion で decode launch 564 -> 452（955）。 | core graph/plan に fused op を持たせ、ggml adapter は同じ adjacency 条件を翻訳。 |
| program/kernel lookup | 1450-1487, 1552-1583 | kernel 名 lookup や `std::map` を launch 時に行う。 | `K_*` index の dense array を backend 内に維持。 |

## 5. Phase 2 / Phase 3 の小コミット案

各 commit 後の Mac 確認: out-of-tree plugin を `GGML_BACKEND_PATH` で読み、`test-backend-ops -b CL11_0` を default / `GGML_CL11_FORCE_GEN=kepler` / `GGML_CL11_FORCE_GEN=fermi`、`adapters/ggml/tools/mvm_check`、`adapters/ggml/tools/attn_check`。TB250 GPU は coordinator 後続確認。

| 順 | Phase | 触るファイル | move | compatibility shim | テスト |
| --- | --- | --- | --- | --- | --- |
| 1 | P2 | `src/backends/opencl11/device.*`, `adapters/ggml/ggml-cl11.cpp`, CMake | `cl11_device_info`, probe, AMD gen, tuning skeleton。 | ggml registry は従来 API 名で backend device list を参照。 | build plugin、`test-backend-ops` 3 configs。 |
| 2 | P2 | `src/backends/opencl11/context.*` | `cl11_ensure_context`, error string, queue/profile flags。 | `cl11_device_context` に opaque backend context pointer を残す。 | 上記 + list devices。 |
| 3 | P2 | `src/backends/opencl11/kernels.*`, generated kernel include CMake | `cl11_kernel_id/table`, build/release/program split/kprogram。 | `ggml-cl11.cpp` の `c->kernels[K_*]` access は inline accessor 経由。 | 上記 + `GGML_CL11_SPLIT_PROGRAMS=1` の subset。 |
| 4 | P2 | `src/backends/opencl11/buffer.*` | `cl11_max_*`, alloc/probe/fill、const sub-buffer cache、release order。 | ggml buffer vtable は `ggml_backend_cl11_*` 名で残し backend buffer handle を保持。 | 上記 + large alloc / strict alloc smoke。 |
| 5 | P2 | `src/backends/opencl11/executor.*` | `cl11_args`, `cl11_launch`, profiling collect/print。 | op functions は既存名のまま executor API を呼ぶ。 | 上記 + `GGML_CL11_PROFILE=1` smoke。 |
| 6 | P3 | `src/core/tensor/*` | `cl11_td`, `cl11_fastdiv_init`, `cl11_bcast`, fit-int validation。 | adapter の `cl11_ref` は ggml tensor -> core desc 変換だけに縮小。 | `test-backend-ops` 3 configs。 |
| 7 | P3 | `src/core/quant/*` | `cl11_rp_*` pack/unpack/geometry、R32 layout enum。 | ggml set/get tensor は core quant API を呼ぶ。 | `test-backend-ops` + partial set/get を含む小テスト追加提案。 |
| 8 | P3 | `src/core/ops/*`, `src/backends/opencl11/ops.*` | `cl11_supports_op_impl` の ggml 非依存条件、`cl11_op_*` launch 群。 | ggml adapter は `GGML_OP_*` を core op に変換し既存 public API を維持。 | `test-backend-ops` 3 configs、`mvm_check`、`attn_check`。 |
| 9 | P3 | `src/core/graph/*`, `adapters/ggml/graph.*` | fusion 判定と execution plan。 | 当面 `ggml_backend_cl11_graph_optimize` は shim として残し、plan 生成へ委譲。 | launch count (`GGML_CL11_LAUNCH_LOG=1`) が増えないことを確認。 |
| 10 | P3 | CMake / docs | `scrapgpu_core`, `scrapgpu_opencl11`, `ggml-cl11` target 分離。 | out-of-tree plugin と in-tree patch の両方で同じ backend objects/static lib を link。 | in-tree build と plugin build の両方。 |

Step 1 done (2026-10-09): moved `cl11_device_info`, device string/scalar helpers, `GGML_CL11_DEVICES` filtering, AMD generation detection, `cl11_tuning_for`, `cl11_config`, `cl11_mmx` parsing/validation and the device probe helper to `src/backends/opencl11/device.{h,cpp}`. `ggml-cl11.cpp` now keeps a registry compatibility shim that materializes the existing ggml `cl11_device_context` objects from backend-probed device info; context/program/buffer/executor state remains in ggml for later steps.

Step 2 done (2026-10-09): moved OpenCL error strings and lazy `cl_context` / in-order `cl_command_queue` creation, including `GGML_CL11_PROFILE` and `GGML_CL11_LAUNCH_LOG` flag capture, to `src/backends/opencl11/context.{h,cpp}`. `ggml-cl11.cpp` keeps a thin `cl11_ensure_context` compatibility shim because later Phase 2 steps still store context, queue, profiling, program, buffer and executor state on the ggml device wrapper.

Step 3 done (2026-10-09): moved the kernel id/class/table helpers and OpenCL program build/source embedding to `src/backends/opencl11/kernels.{h,cpp}`, and moved the `.cl` files unchanged to `src/backends/opencl11/kernels/`. `ggml-cl11.cpp` keeps program orchestration compatibility shims for work-group reduction, split-program selection, K-quant program creation and kernel handle storage until executor/buffer state moves.
`GGML_CL11_PRINT_BUILD_OPTS=1` prints the exact OpenCL build options string before `clBuildProgram` for byte-for-byte comparison.

Step 4 done (2026-10-09): moved buffer limits/alignment, strict/above-max allocation policy, end-of-buffer allocation probe, constant sub-buffer cache creation and newest-first release helper to `src/backends/opencl11/buffer.{h,cpp}`. `ggml-cl11.cpp` keeps the ggml buffer vtable, dummy base pointer/offset handling, R32 repack transfer and the `cl11_fill` launch shim until the executor moves in Step 5. The kernel file list and generated-header embedding loop now live once in `src/backends/opencl11/opencl11_kernels.cmake`, included by both root and adapter CMake paths.

## 6. リスクと未決

| 項目 | リスク | 未決 / 提案 |
| --- | --- | --- |
| out-of-tree plugin と in-tree patch | `adapters/ggml/CMakeLists.txt` は 2 経路を持つ（1-12, 28-117）。分離後 target 名や include path がずれると片方だけ壊れる。 | 共通 object/static lib を作り、最後の `ggml-cl11` target だけ経路差を持つ提案。 |
| symbol visibility | `libggml-cl11.so` 内に新 static libraries を link する場合、OpenCL backend の内部 symbol が外へ漏れる/消える可能性。 | public は `ggml_backend_cl11_*` と将来 `scrapgpu_*` C ABI のみに限定。internal は hidden visibility を提案。 |
| core が OpenCL なしで build | 現 CMake は常に `find_package(OpenCL REQUIRED)`（24）。 | `scrapgpu_core` target は OpenCL include/link なし。`scrapgpu_opencl11` のみ OpenCL required。 |
| core が ggml なしで build | 現 cpp は `ggml-cl11.h`, `ggml-impl.h`, `ggml-backend-impl.h` を直接 include（25-29）。 | core headers/sources から ggml include を禁止し、adapter target だけ ggml を include。 |
| kernel source embedding | CMake が `.cl` を header に埋め込む（127-177）。backend 分離で生成 header の場所が変わる。 | kernel embedding は `backends/opencl11` target 所有に移す。`GGML_CL11_KERNEL_DIR` は backend debug option として維持。 |
| lifetime | context/program/cache は device context に混在し、ggml dev/buft も同 struct に入る（395-497）。 | backend-owned context と adapter-owned vtable object の owner を分け、release order を単体テスト化する提案。 |
| fusion planner | ggml graph の adjacency と mutable reorder に依存（3551-3796）。 | core plan API ができるまで、fusion 判定は adapter に残す。移行後も launch count を比較する。 |
| 性能 ±3% | `docs/plans/V0.2_SPLIT_BRIEF.md` は Gate 2 で性能 ±3% 目標（108-112）。抽象化で launch 前 overhead が増えると decode に直撃。 | per-node plan cache、dense kernel array、allocation-free launch を受け入れ条件にする。 |
