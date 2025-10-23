/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "session/output_driver.h"

//#include "util/function.h"
#include "util/image.h"
#include "util/string.h"
#include "util/unique_ptr.h"
#include "util/vector.h"
#include "util/thread.h"

CCL_NAMESPACE_BEGIN

class FrameOutputDriver : public OutputDriver {
public:
	using LogFunction = std::function<void(const string&)>;

	FrameOutputDriver(const string_view pass, LogFunction log);
	virtual ~FrameOutputDriver();

	void write_render_tile(const Tile& tile) override;

	void renderBegin();
	void renderEnd();

	void wait();
	bool ready() const;

public:
	vector<half4> pixels;
	void* d_pixels;
	bool render_finished;
	//std::atomic<bool> render_finished;
	thread_mutex mutex;
	thread_condition_variable cv;
	std::chrono::time_point<std::chrono::steady_clock> start;
	float duration;

protected:
	string pass_;
	LogFunction log_;
};

CCL_NAMESPACE_END
