#include "stereotgv.h"

__global__ void TgvUpdatePrimalVariablesKernel(float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float mu,
	float* u, float2* v,
	float* u_s, float2* v_s,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	float desiredRadius = (float)width / 2.20f;
	float halfWidth = (float)width / 2.0f;
	float halfHeight = (float)height / 2.0f;
	float radius = sqrtf((iy - halfHeight) * (iy - halfHeight) + (ix - halfWidth) * (ix - halfWidth));

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		if (radius >= desiredRadius)
		{
			u_s[pos] = 0.0f;
			v_s[pos] = make_float2(0.0f, 0.0f);
			//u_s[pos] = u_[pos];
			//v_s[pos] = v_[pos];
		}
		else {
			int right = (ix + 1) + iy * stride;
			int down = ix + (iy + 1) * stride;
			int left = (ix - 1) + iy * stride;
			int up = ix + (iy - 1) * stride;

			float4 qc;
			if (ix == width - 1) {
				qc.x = 0.0f;
				qc.w = 0.0f;
			}
			else {
				qc.x = q[pos].x;
				qc.w = q[pos].w;
			}
			if (iy == height - 1) {
				qc.y = 0.0f;
				qc.z = 0.0f;
			}
			else {
				qc.y = q[pos].y;
				qc.z = q[pos].z;
			}

			//qw_x = [zeros(M, 1, 1), q(:, 1 : end - 1, 1)];
			//qw_w = [zeros(M, 1, 1), q(:, 1 : end - 1, 4)];
			float qw_x, qw_w;
			if ((ix - 1) >= 0) {
				qw_x = q[left].x;
				qw_w = q[left].w;
			}
			else {
				qw_x = 0.0f;
				qw_w = 0.0f;
			}

			//qn_y = [zeros(1, N, 1); q(1:end - 1, : , 2)];
			//qn_z = [zeros(1, N, 1); q(1:end - 1, : , 3)];
			float qn_y, qn_z;
			if ((iy - 1) >= 0) {
				qn_y = q[up].y;
				qn_z = q[up].z;
			}
			else {
				qn_y = 0.0f;
				qn_z = 0.0f;
			}

			//div_q(:, : , 1) = (qc(:, : , 1) - qw_x) + (qc(:, : , 3) - qn_z);
			//div_q(:, : , 2) = (qc(:, : , 4) - qw_w) + (qc(:, : , 2) - qn_y);
			float2 div_q;
			div_q.x = (qc.x - qw_x) + (qc.z - qn_z);
			div_q.y = (qc.w - qw_w) + (qc.y - qn_y);

			//dq_tensor(:, : , 1) = a.*p(:, : , 1) + c.*p(:, : , 2);
			//dq_tensor(:, : , 2) = c.*p(:, : , 1) + b.*p(:, : , 2);
			float2 dq_tensor;
			float2 ppos = p[pos];
			dq_tensor.x = a[pos] * ppos.x + c[pos] * ppos.y;
			dq_tensor.y = c[pos] * ppos.x + b[pos] * ppos.y;

			float eta_v1s = eta_v1[pos];
			float eta_v2s = eta_v2[pos];
			float2 v_pos = v_[pos];
			float2 vpos;
			if (eta_v1s != 0) {
				vpos.x = v_pos.x + (tau / eta_v1s) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
			}
			else {
				vpos.x = v_pos.x + (tau) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
			}
			if (eta_v2s != 0) {
				vpos.y = v_pos.y + (tau / eta_v2s) * (alpha1 * dq_tensor.y + alpha0 * div_q.y);
			}
			else {
				vpos.x = v_pos.x + (tau) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
			}
			v[pos] = vpos;

			u_s[pos] = u[pos] + mu * (u[pos] - u_[pos]);
			v_s[pos].x = vpos.x + mu * (vpos.x - v_pos.x);
			v_s[pos].y = vpos.y + mu * (vpos.y - v_pos.y);
		}
	}
}

void StereoTgv::UpdatePrimalVariables(float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float mu,
	float* u, float2* v,
	float* u_s, float2* v_s,
	int w, int h, int s) 
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	TgvUpdatePrimalVariablesKernel << < blocks, threads >> > (u_, v_, p, q,
		a, b, c, 
		tau, eta_v1, eta_v2, 
		alpha0, alpha1, mu,
		u, v, 
		u_s, v_s,
		w, h, s);
}


// **********************
// MASKED
// **********************
__global__ 
void TgvUpdatePrimalVariablesMaskedKernel(float* mask, float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float mu,
	float* u, float2* v,
	float* u_s, float2* v_s,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	int right = (ix + 1) + iy * stride;
	int down = ix + (iy + 1) * stride;
	int left = (ix - 1) + iy * stride;
	int up = ix + (iy - 1) * stride;

	float maskRight, maskLeft, maskUp, maskDown;
	if (ix + 1 >= width) maskRight = 0.0f;
	else maskRight = mask[right];
	if (ix - 1 < 0) maskLeft = 0.0f;
	else maskLeft = mask[left];
	if (iy + 1 >= height) maskDown = 0.0f;
	else maskDown = mask[down];
	if (iy - 1 < 0) maskUp = 0.0f;
	else maskUp = mask[up];

	float4 qc;
	if (maskRight == 0.0f) {
		qc.x = 0.0f;
		qc.w = 0.0f;
	}
	else {
		qc.x = q[pos].x;
		qc.w = q[pos].w;
	}
	if (maskDown == 0.0f) {
		qc.y = 0.0f;
		qc.z = 0.0f;
	}
	else {
		qc.y = q[pos].y;
		qc.z = q[pos].z;
	}

	//qw_x = [zeros(M, 1, 1), q(:, 1 : end - 1, 1)];
	//qw_w = [zeros(M, 1, 1), q(:, 1 : end - 1, 4)];
	float qw_x, qw_w;
	if (maskLeft != 0.0f) {
		qw_x = q[left].x;
		qw_w = q[left].w;
	}
	else {
		qw_x = 0.0f;
		qw_w = 0.0f;
	}

	//qn_y = [zeros(1, N, 1); q(1:end - 1, : , 2)];
	//qn_z = [zeros(1, N, 1); q(1:end - 1, : , 3)];
	float qn_y, qn_z;
	if (maskUp != 0.0f) {
		qn_y = q[up].y;
		qn_z = q[up].z;
	}
	else {
		qn_y = 0.0f;
		qn_z = 0.0f;
	}

	//div_q(:, : , 1) = (qc(:, : , 1) - qw_x) + (qc(:, : , 3) - qn_z);
	//div_q(:, : , 2) = (qc(:, : , 4) - qw_w) + (qc(:, : , 2) - qn_y);
	float2 div_q;
	div_q.x = (qc.x - qw_x) + (qc.z - qn_z);
	div_q.y = (qc.w - qw_w) + (qc.y - qn_y);

	//dq_tensor(:, : , 1) = a.*p(:, : , 1) + c.*p(:, : , 2);
	//dq_tensor(:, : , 2) = c.*p(:, : , 1) + b.*p(:, : , 2);
	float2 dq_tensor;
	float2 ppos = p[pos];
	dq_tensor.x = a[pos] * ppos.x + c[pos] * ppos.y;
	dq_tensor.y = c[pos] * ppos.x + b[pos] * ppos.y;

	float eta_v1s = eta_v1[pos];
	float eta_v2s = eta_v2[pos];
	float2 v_pos = v_[pos];
	float2 vpos;
	if (eta_v1s != 0) {
		vpos.x = v_pos.x + (tau / eta_v1s) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
	}
	else {
		vpos.x = v_pos.x + (tau) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
	}
	if (eta_v2s != 0) {
		vpos.y = v_pos.y + (tau / eta_v2s) * (alpha1 * dq_tensor.y + alpha0 * div_q.y);
	}
	else {
		vpos.x = v_pos.x + (tau) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
	}
	v[pos] = vpos;

	u_s[pos] = u[pos] + mu * (u[pos] - u_[pos]);
	v_s[pos].x = vpos.x + mu * (vpos.x - v_pos.x);
	v_s[pos].y = vpos.y + mu * (vpos.y - v_pos.y);
}

void StereoTgv::UpdatePrimalVariablesMasked(float * mask, float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float mu,
	float* u, float2* v,
	float* u_s, float2* v_s,
	int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	TgvUpdatePrimalVariablesMaskedKernel << < blocks, threads >> > (mask, u_, v_, p, q,
		a, b, c,
		tau, eta_v1, eta_v2,
		alpha0, alpha1, mu,
		u, v,
		u_s, v_s,
		w, h, s);
}