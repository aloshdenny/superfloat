module ActivationUnit (
    input wire clk,
    input wire start,             // Start processing
    input wire select,            // 0: Linear, 1: ReLU
    input wire [15:0] buffer,     // 16-bit buffer holding matrix values
    output reg [15:0] result      // Output after activation
);
    reg signed [15:0] data;   // Assuming 16-bit fixed/floating point numbers
    reg signed [15:0] output_data;

    always @(posedge clk) begin
        if (start) begin
            // Extract value from the buffer
            data = buffer;
            
            // Apply activation function
            if (select == 1'b1) begin
                // ReLU: Max(0, value)
                output_data = (data < 0) ? 0 : data;
            end else begin
                // Linear: Identity function
                output_data = data;
            end

            // Store result
            result = output_data;
        end
    end
endmodule
